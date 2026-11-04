"""Lightweight, flow-sensitive dataflow over one Python module / notebook.

This is deliberately simple. The whole file is treated as one scope and
statements are processed in source order (a notebook "run all"). For every
variable we track a *role*:

* ``FULL``  - comes from the whole dataset (read_csv, load_*, ...) or is derived
  from it before any split.
* ``TRAIN`` / ``VALID`` / ``TEST`` - comes from one side of a split
  (train_test_split outputs, CV fold indices, date/positional slices, or a
  variable whose *name* says train/val/test).
* ``UNKNOWN`` - anything else.

Roles combine as TEST > VALID > TRAIN > FULL > UNKNOWN, so ``X.iloc[test_idx]``
is TEST and ``scaler.transform(X_train)`` is TRAIN.

We also record every call (with alias-resolved names), every split event,
every ``.fit``-style call (with the estimator class, when known) and every
assignment. Rules query these facts; they never re-walk the AST unless they
need something special.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum

from .source import SourceFile


class Role(str, Enum):
    UNKNOWN = "unknown"
    FULL = "full"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    @property
    def rank(self) -> int:
        return _ROLE_RANK[self]

    @property
    def is_split_side(self) -> bool:
        return self in (Role.TRAIN, Role.VALID, Role.TEST)


_ROLE_RANK = {Role.UNKNOWN: 0, Role.FULL: 1, Role.TRAIN: 2, Role.VALID: 3, Role.TEST: 4}


def combine(roles: Iterable[Role]) -> Role:
    best = Role.UNKNOWN
    for r in roles:
        if r.rank > best.rank:
            best = r
    return best


# --------------------------------------------------------------------------
# Knowledge base: names of things we recognise. Short (unqualified) names.
# --------------------------------------------------------------------------

DATA_LOADERS = {
    "read_csv", "read_parquet", "read_excel", "read_json", "read_sql", "read_sql_query",
    "read_sql_table", "read_table", "read_feather", "read_pickle", "read_hdf", "read_orc",
    "read_stata", "read_sas", "read_spss", "read_fwf", "load", "loadtxt", "genfromtxt",
    "load_dataset", "read_ipc", "scan_csv", "scan_parquet", "DataFrame",
}
DATA_LOADER_PREFIXES = ("load_", "fetch_", "make_")  # sklearn.datasets

SPLIT_FUNCS = {"train_test_split"}
CV_SPLITTERS = {
    "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "RepeatedKFold",
    "RepeatedStratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit",
    "TimeSeriesSplit", "LeaveOneOut", "LeavePOut", "LeaveOneGroupOut", "LeavePGroupsOut",
    "PredefinedSplit",
}
CV_EVAL_FUNCS = {
    "cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
    "validation_curve", "permutation_test_score",
}
SEARCH_CV = {
    "GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV",
    "BayesSearchCV", "TuneSearchCV",
}
PREPROCESSORS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "QuantileTransformer", "PowerTransformer", "SimpleImputer", "KNNImputer",
    "IterativeImputer", "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "LabelBinarizer",
    "MultiLabelBinarizer", "PCA", "TruncatedSVD", "KernelPCA", "IncrementalPCA", "FastICA",
    "NMF", "FactorAnalysis", "KBinsDiscretizer", "TfidfVectorizer", "CountVectorizer",
    "HashingVectorizer", "TfidfTransformer", "SplineTransformer", "TSNE", "UMAP",
    "Isomap", "LocallyLinearEmbedding", "ColumnTransformer",
}
SELECTORS = {
    "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe", "RFE", "RFECV",
    "SelectFromModel", "SequentialFeatureSelector", "GenericUnivariateSelect", "Boruta",
    "BorutaPy",
}
SAMPLERS = {
    "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN",
    "RandomOverSampler", "RandomUnderSampler", "SMOTETomek", "SMOTEENN", "NearMiss",
    "TomekLinks", "EditedNearestNeighbours", "ClusterCentroids",
}
TARGET_ENCODERS = {
    "TargetEncoder", "MEstimateEncoder", "CatBoostEncoder", "LeaveOneOutEncoder",
    "JamesSteinEncoder", "WOEEncoder", "GLMMEncoder", "QuantileEncoder",
}
PIPELINE_BUILDERS = {"Pipeline", "make_pipeline", "ImbPipeline"}
FIT_METHODS = {"fit", "fit_transform", "fit_resample", "fit_predict", "partial_fit", "fit_sample"}

_NAME_TOKENS = {
    Role.TRAIN: {"train", "training", "trn", "tr"},
    Role.VALID: {"val", "va", "valid", "validation", "dev", "eval"},
    Role.TEST: {"test", "testing", "tst", "holdout", "te", "oos"},
}
_TOKEN_SPLIT = re.compile(r"[_\W\d]+")
_NOT_DATA_TOKENS = {"tf", "transform", "transforms", "aug", "augment", "metric", "metrics", "size", "ratio", "frac"}

_DATE_WORDS = re.compile(r"(date|time|timestamp|datetime|month|year|day|week|period|dt$|^ts$)", re.I)


def role_from_name(name: str) -> Role:
    """Guess a role from a variable name, e.g. ``X_test`` -> TEST."""
    tokens = [t for t in _TOKEN_SPLIT.split(name.lower()) if t]
    if any(t in _NOT_DATA_TOKENS for t in tokens):
        return Role.UNKNOWN  # tf_test, test_transform, eval_metric ... are not data
    found = Role.UNKNOWN
    for role, words in _NAME_TOKENS.items():
        if any(t in words for t in tokens):
            found = combine([found, role])
    return found


def looks_like_date(text: str) -> bool:
    return bool(_DATE_WORDS.search(text))


def is_estimator_like(cls: str | None) -> bool:
    return bool(cls) and cls[0].isupper()  # type: ignore[index]


# --------------------------------------------------------------------------
# Fact records
# --------------------------------------------------------------------------


@dataclass
class CallInfo:
    node: ast.Call
    line: int
    name: str  # alias-resolved dotted name, e.g. "sklearn.model_selection.train_test_split", or "df.fillna"
    func: str  # last component, e.g. "train_test_split" / "fillna"
    receiver: str | None  # for ``x.method(...)`` where x is a plain name: "x"
    receiver_expr: ast.expr | None  # the expression before ``.method`` (any shape)
    args: list[ast.expr]
    kwargs: dict[str, ast.expr]
    stmt: ast.stmt
    loop_depth: int  # how many for/while loops enclose this call
    func_scope: str | None  # enclosing function name, if any

    def kwarg_const(self, key: str) -> object:
        """Literal value of a keyword argument, or ``None`` if absent / not literal."""
        node = self.kwargs.get(key)
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    def has_kwarg(self, key: str) -> bool:
        return key in self.kwargs


@dataclass
class SplitEvent:
    line: int
    kind: str  # "train_test_split" | "cv_split" | "subset" | "time_slice" | "positional" | "random_sample"
    node: ast.AST
    call: CallInfo | None
    inputs: list[str]
    train: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    splitter: str | None = None  # CV splitter class for cv_split, e.g. "KFold"

    @property
    def is_random(self) -> bool:
        if self.kind == "train_test_split" and self.call is not None:
            return self.call.kwarg_const("shuffle") is not False
        if self.kind == "random_sample":
            return True
        if self.kind == "cv_split":
            return self.splitter not in {"TimeSeriesSplit", "PredefinedSplit"}
        return False


@dataclass
class FitCall:
    call: CallInfo
    method: str  # "fit", "fit_transform", ...
    estimator: str | None  # class short name, e.g. "StandardScaler" (None if unknown)
    receiver: str | None  # variable the method is called on
    data_args: list[ast.expr]  # positional args + X=/y= kwargs
    roles: list[Role]  # role of each data arg at the call
    line: int

    @property
    def role(self) -> Role:
        return combine(self.roles)

    @property
    def x_role(self) -> Role:
        return self.roles[0] if self.roles else Role.UNKNOWN

    @property
    def has_y(self) -> bool:
        return len(self.data_args) >= 2


@dataclass
class Assignment:
    line: int
    targets: list[str]  # plain names bound (tuple targets flattened)
    value: ast.expr
    role: Role  # role assigned to the targets
    node: ast.stmt
    subscript_base: str | None = None  # for ``df["col"] = ...``: "df"
    subscript_key: str | None = None  # literal column key, if any


@dataclass
class EstimatorDef:
    line: int
    cls: str  # short class name ("Pipeline" for make_pipeline)
    call: ast.Call
    steps: list[str] = field(default_factory=list)  # step classes for pipelines/ColumnTransformer


# --------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------


class Analysis:
    """All facts about one file. Build with ``Analysis(source_file)``."""

    def __init__(self, src: SourceFile) -> None:
        self.src = src
        self.tree = src.tree
        self.imports: dict[str, str] = {}
        self.calls: list[CallInfo] = []
        self.splits: list[SplitEvent] = []
        self.cv_evals: list[CallInfo] = []  # cross_val_score(...) & co, and SearchCV(...).fit
        self.fits: list[FitCall] = []
        self.assignments: list[Assignment] = []
        self.estimators: dict[str, list[EstimatorDef]] = {}
        self.parents: dict[ast.AST, ast.AST] = {}
        self.target_columns: dict[str, int] = {}  # column literal -> line where used as y
        self.target_vars: set[str] = set()  # variables holding the target (y)
        self.datetime_columns: set[str] = set()
        self.datetime_evidence: list[tuple[int, str]] = []
        self.string_constants: set[str] = set()
        self._history: dict[str, list[tuple[int, Role]]] = {}
        self._stmt_of: dict[ast.AST, ast.stmt] = {}
        self._build()

    # ------------------------------------------------------------- queries

    @property
    def first_split_line(self) -> int | None:
        lines = [s.line for s in self.splits]
        return min(lines) if lines else None

    @property
    def has_split(self) -> bool:
        return bool(self.splits)

    def split_after(self, line: int) -> SplitEvent | None:
        """First split event strictly after ``line`` (None if none)."""
        later = [s for s in self.splits if s.line > line]
        return min(later, key=lambda s: s.line) if later else None

    def split_before(self, line: int) -> SplitEvent | None:
        earlier = [s for s in self.splits if s.line < line]
        return max(earlier, key=lambda s: s.line) if earlier else None

    def role_at(self, name: str, line: int) -> Role:
        """Role of variable ``name`` as seen by a statement on ``line``."""
        hist = self._history.get(name)
        if hist:
            for ln, role in reversed(hist):
                if ln < line:
                    return role
            # only assigned later (or on this line): fall through to name hint
        return role_from_name(name)

    def role_of(self, expr: ast.AST | None, line: int | None = None) -> Role:
        """Combined role of every data variable referenced in ``expr``."""
        if expr is None:
            return Role.UNKNOWN
        ln = line if line is not None else getattr(expr, "lineno", 10**9)
        roles = [self.role_at(n, ln) for n in self.data_names_in(expr)]
        return combine(roles)

    def data_names_in(self, expr: ast.AST) -> list[str]:
        """Plain variable names referenced in ``expr`` (excluding called functions and modules)."""
        called = set()
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    called.add(id(f))
                elif isinstance(f, ast.Attribute):
                    # the receiver *is* data (df.fillna) unless it is a module alias
                    base = f.value
                    if isinstance(base, ast.Name) and base.id in self.imports:
                        called.add(id(base))
        out = []
        for n in ast.walk(expr):
            if isinstance(n, ast.Name) and id(n) not in called and n.id not in self.imports:
                if self.estimator_class(n.id, getattr(n, "lineno", 10**9) + 1) is None:
                    out.append(n.id)
        return out

    def estimator_def(self, name: str, line: int) -> EstimatorDef | None:
        defs = self.estimators.get(name)
        if not defs:
            return None
        best = None
        for d in defs:
            if d.line <= line:
                best = d
        return best or defs[0]

    def estimator_class(self, name: str, line: int) -> str | None:
        d = self.estimator_def(name, line)
        return d.cls if d else None

    def resolve(self, node: ast.AST | None) -> str | None:
        """Dotted name of a Name/Attribute chain with import aliases expanded."""
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            head = self.imports.get(cur.id, cur.id)
            parts.append(head)
        elif isinstance(cur, ast.Call):
            inner = self.resolve(cur.func)
            parts.append((inner or "<call>") + "()")
        elif cur is not None and parts:
            parts.append("<expr>")
        else:
            return None
        return ".".join(reversed(parts))

    def calls_named(self, *funcs: str) -> Iterator[CallInfo]:
        """Calls whose last name component is in ``funcs``."""
        wanted = set(funcs)
        return (c for c in self.calls if c.func in wanted)

    def stmt_of(self, node: ast.AST) -> ast.stmt | None:
        return self._stmt_of.get(node)

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(node)

    def is_before_any_split(self, line: int) -> bool:
        first = self.first_split_line
        return first is not None and line < first

    # ------------------------------------------------------------- building

    def _build(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) < 80:
                    self.string_constants.add(node.value)
        self._collect_imports()
        _Walker(self).visit(self.tree)
        self._post()

    def _collect_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{node.module}.{a.name}"

    def _set_role(self, name: str, line: int, role: Role) -> None:
        self._history.setdefault(name, []).append((line, role))

    def _post(self) -> None:
        for c in self.calls:
            # datetime evidence
            if c.func == "to_datetime":
                for a in c.args:
                    key = _subscript_key(a)
                    if key:
                        self.datetime_columns.add(key)
                self.datetime_evidence.append((c.line, "pd.to_datetime(...)"))
            if c.func in {"read_csv", "read_excel", "read_table"} and "parse_dates" in c.kwargs:
                self.datetime_evidence.append((c.line, "read_csv(parse_dates=...)"))
                try:
                    val = ast.literal_eval(c.kwargs["parse_dates"])
                    if isinstance(val, (list, tuple)):
                        self.datetime_columns.update(str(v) for v in val)
                except Exception:
                    pass
            if c.func in {"DatetimeIndex", "date_range", "to_period", "resample"}:
                self.datetime_evidence.append((c.line, f"{c.func}(...)"))
            if c.func == "set_index" and c.args:
                key = _const_str(c.args[0])
                if key and looks_like_date(key):
                    self.datetime_evidence.append((c.line, f"set_index({key!r})"))
                    self.datetime_columns.add(key)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Attribute) and n.attr == "dt" and isinstance(n.ctx, ast.Load):
                self.datetime_evidence.append((n.lineno, ".dt accessor"))
                key = _subscript_key(n.value)
                if key:
                    self.datetime_columns.add(key)


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _subscript_key(node: ast.AST | None) -> str | None:
    """``df["col"]`` -> "col";  ``df.col`` -> "col"."""
    if isinstance(node, ast.Subscript):
        return _const_str(node.slice)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            out.extend(_target_names(e.value if isinstance(e, ast.Starred) else e))
        return out
    return []


def _flat_targets(t: ast.AST) -> list[ast.AST]:
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[ast.AST] = []
        for e in t.elts:
            out.extend(_flat_targets(e))
        return out
    return [t]


class _Walker(ast.NodeVisitor):
    def __init__(self, a: Analysis) -> None:
        self.a = a
        self.loop_depth = 0
        self.func_stack: list[str] = []
        self.stmt_stack: list[ast.stmt] = []

    # --- structure ---------------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            self.stmt_stack.append(node)
            super().generic_visit(node)
            self.stmt_stack.pop()
        else:
            if self.stmt_stack:
                self.a._stmt_of[node] = self.stmt_stack[-1]
            super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.func_stack.append(node.name)
        # parameters named like data get a name-based role only (handled by role_from_name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.stmt_stack.append(node)
        self.visit(node.iter)
        self._handle_for_target(node)
        self.loop_depth += 1
        for s in node.body + node.orelse:
            self.visit(s)
        self.loop_depth -= 1
        self.stmt_stack.pop()

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def visit_While(self, node: ast.While) -> None:
        self.stmt_stack.append(node)
        self.visit(node.test)
        self.loop_depth += 1
        for s in node.body + node.orelse:
            self.visit(s)
        self.loop_depth -= 1
        self.stmt_stack.pop()

    # --- calls ---------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        a = self.a
        if self.stmt_stack:
            a._stmt_of[node] = self.stmt_stack[-1]
        name = a.resolve(node.func) or "<call>"
        func = name.rsplit(".", 1)[-1]
        receiver = None
        receiver_expr = None
        if isinstance(node.func, ast.Attribute):
            receiver_expr = node.func.value
            if isinstance(receiver_expr, ast.Name) and receiver_expr.id not in a.imports:
                receiver = receiver_expr.id
        info = CallInfo(
            node=node,
            line=node.lineno,
            name=name,
            func=func,
            receiver=receiver,
            receiver_expr=receiver_expr,
            args=list(node.args),
            kwargs={k.arg: k.value for k in node.keywords if k.arg},
            stmt=self.stmt_stack[-1] if self.stmt_stack else node,  # type: ignore[arg-type]
            loop_depth=self.loop_depth,
            func_scope=self.func_stack[-1] if self.func_stack else None,
        )
        # visit children first so nested calls are recorded in order
        self.generic_visit(node)
        a.calls.append(info)

        if func in CV_EVAL_FUNCS:
            a.cv_evals.append(info)
        if func in FIT_METHODS and isinstance(node.func, ast.Attribute):
            self._record_fit(info)
        if func in {"pop"} and node.args:
            pass

    def _record_fit(self, info: CallInfo) -> None:
        a = self.a
        recv = info.receiver_expr
        est: str | None = None
        if info.receiver:
            est = a.estimator_class(info.receiver, info.line)
        elif isinstance(recv, ast.Call):
            short = (a.resolve(recv.func) or "").rsplit(".", 1)[-1]
            if is_estimator_like(short):
                est = "Pipeline" if short in PIPELINE_BUILDERS else short
        data_args = list(info.args)
        for k in ("X", "y", "x", "Y", "X_train", "y_train"):
            if k in info.kwargs:
                data_args.append(info.kwargs[k])
        roles = [a.role_of(arg, info.line) for arg in data_args]
        fit = FitCall(
            call=info,
            method=info.func,
            estimator=est,
            receiver=info.receiver,
            data_args=data_args,
            roles=roles,
            line=info.line,
        )
        a.fits.append(fit)
        if est in SEARCH_CV:
            a.cv_evals.append(info)

    # --- assignments -------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.stmt_stack.append(node)
        self.visit(node.value)
        for t in node.targets:
            self._assign(t, node.value, node)
        self.stmt_stack.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.stmt_stack.append(node)
        if node.value is not None:
            self.visit(node.value)
            self._assign(node.target, node.value, node)
        self.stmt_stack.pop()

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.stmt_stack.append(node)
        self.visit(node.value)
        self.stmt_stack.pop()

    def _assign(self, target: ast.AST, value: ast.expr, stmt: ast.stmt) -> None:
        a = self.a
        line = stmt.lineno
        names = _target_names(target)

        # df["col"] = ...  (mutation of an existing frame)
        if isinstance(target, ast.Subscript):
            base = target.value
            base_name = base.id if isinstance(base, ast.Name) else None
            a.assignments.append(
                Assignment(
                    line=line, targets=[], value=value, role=a.role_of(value, line), node=stmt,
                    subscript_base=base_name, subscript_key=_const_str(target.slice),
                )
            )
            if base_name:
                role = combine([a.role_at(base_name, line), a.role_of(value, line)])
                # a mutation keeps the frame's own role unless it was unknown
                a._set_role(base_name, line, role if a.role_at(base_name, line) is Role.UNKNOWN else a.role_at(base_name, line))
            return

        call = value if isinstance(value, ast.Call) else None
        short = (a.resolve(call.func) or "").rsplit(".", 1)[-1] if call else ""

        # ---- estimator construction: scaler = StandardScaler()
        if call is not None and len(names) == 1 and (is_estimator_like(short) or short in PIPELINE_BUILDERS) \
                and short not in DATA_LOADERS:
            cls = "Pipeline" if short in PIPELINE_BUILDERS else short
            a.estimators.setdefault(names[0], []).append(
                EstimatorDef(line=line, cls=cls, call=call, steps=self._steps(call))
            )
            a.assignments.append(Assignment(line, names, value, Role.UNKNOWN, stmt))
            a._set_role(names[0], line, Role.UNKNOWN)
            return

        # ---- model = est.fit(X, y): alias of the estimator
        if call is not None and short in {"fit", "partial_fit"} and isinstance(call.func, ast.Attribute) \
                and len(names) == 1:
            recv = call.func.value
            if isinstance(recv, ast.Name):
                d = a.estimator_def(recv.id, line)
                if d:
                    a.estimators.setdefault(names[0], []).append(d)
            a.assignments.append(Assignment(line, names, value, Role.UNKNOWN, stmt))
            a._set_role(names[0], line, Role.UNKNOWN)
            return

        # ---- train_test_split
        if call is not None and short in SPLIT_FUNCS:
            inputs = [n.id for n in call.args if isinstance(n, ast.Name)]
            flat = _flat_targets(target)
            out_names = [t.id if isinstance(t, ast.Name) else "" for t in flat]
            ev = SplitEvent(line=line, kind="train_test_split", node=call, call=self._call_info(call),
                            inputs=inputs)
            n_in = len(call.args)
            if n_in and len(out_names) == 2 * n_in:
                for i, nm in enumerate(out_names):
                    if not nm:
                        continue
                    hinted = role_from_name(nm)
                    pos_role = Role.TRAIN if i % 2 == 0 else Role.TEST
                    role = hinted if hinted.is_split_side else pos_role
                    self._bind_side(ev, nm, role, line)
            else:
                for nm in out_names:
                    if nm:
                        r = role_from_name(nm)
                        self._bind_side(ev, nm, r if r.is_split_side else Role.TRAIN, line)
            a.splits.append(ev)
            a.assignments.append(Assignment(line, names, value, Role.UNKNOWN, stmt))
            return

        role = a.role_of(value, line)
        # data loaders produce FULL data (also inside a chain: pd.read_csv(...).drop_duplicates())
        loader_calls = [
            (a.resolve(n.func) or "").rsplit(".", 1)[-1] for n in ast.walk(value) if isinstance(n, ast.Call)
        ]
        if any(((s in DATA_LOADERS or s.startswith(DATA_LOADER_PREFIXES)) and s != "load") or (s == "load"
               and isinstance(value, ast.Call)) for s in loader_calls):
            role = combine([role, Role.FULL])
            if short.startswith(("load_", "fetch_", "make_")) and len(names) == 2:
                # X, y = load_iris(return_X_y=True)
                a.target_vars.add(names[1])

        # target extraction: y = df["target"] / df.pop("target") / df.target
        if len(names) == 1:
            self._maybe_target(names[0], value, line)

        # tuple unpack of a non-split call: give everyone the combined role
        for nm in names:
            hinted = role_from_name(nm)
            final = role
            if hinted.is_split_side and role in (Role.FULL, Role.UNKNOWN):
                # train = df[df.date < "2020"]  -> a subset split
                if role is Role.FULL:
                    self._record_subset(nm, hinted, value, line)
                final = hinted
            elif hinted in (Role.VALID, Role.TEST) and role is Role.TRAIN:
                # X_val = X_train[...]  -> a nested split
                self._record_subset(nm, hinted, value, line)
                final = hinted
            a._set_role(nm, line, final)
        a.assignments.append(Assignment(line, names, value, role, stmt))

    # --- helpers -----------------------------------------------------------

    def _bind_side(self, ev: SplitEvent, name: str, role: Role, line: int) -> None:
        {Role.TRAIN: ev.train, Role.TEST: ev.test, Role.VALID: ev.valid}.get(role, ev.train).append(name)
        self.a._set_role(name, line, role if role.is_split_side else Role.TRAIN)

    def _call_info(self, call: ast.Call) -> CallInfo | None:
        for c in reversed(self.a.calls):
            if c.node is call:
                return c
        return None

    def _record_subset(self, name: str, role: Role, value: ast.expr, line: int) -> None:
        a = self.a
        kind = "subset"
        text = ast.unparse(value) if hasattr(ast, "unparse") else ""
        has_compare = any(isinstance(n, ast.Compare) for n in ast.walk(value))
        has_slice = any(isinstance(n, ast.Slice) for n in ast.walk(value))
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sample"
               for n in ast.walk(value)):
            kind = "random_sample"
        elif has_compare and (looks_like_date(text) or any(
                isinstance(n, ast.Constant) and isinstance(n.value, str) and re.match(r"\d{4}-\d\d", n.value)
                for n in ast.walk(value))):
            kind = "time_slice"
        elif has_slice:
            kind = "positional"
        inputs = [n for n in a.data_names_in(value) if a.role_at(n, line) is Role.FULL]
        ev = SplitEvent(line=line, kind=kind, node=value, call=None, inputs=inputs)
        {Role.TRAIN: ev.train, Role.TEST: ev.test, Role.VALID: ev.valid}[role].append(name)
        # merge with a split on the same/adjacent line of the same kind (train=..; test=..)
        for prev in a.splits:
            if prev.kind == kind and prev.call is None and abs(prev.line - line) <= 3 and prev.inputs == inputs:
                prev.train += ev.train
                prev.test += ev.test
                prev.valid += ev.valid
                return
        a.splits.append(ev)

    def _maybe_target(self, name: str, value: ast.expr, line: int) -> None:
        a = self.a
        key = None
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            key = _const_str(value.slice)
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "pop" \
                and value.args:
            key = _const_str(value.args[0])
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and \
                a.role_at(value.value.id, line) is Role.FULL and value.attr not in {"values", "index", "columns", "shape", "T"}:
            key = value.attr
        lname = name.lower()
        is_y = lname in {"y", "target", "labels", "label", "y_all", "y_full", "yy"} or lname.startswith("y_") or lname.startswith("target")
        if is_y:
            a.target_vars.add(name)
            if key:
                a.target_columns.setdefault(key, line)
        elif key and lname in {"y_train", "ytrain"}:
            a.target_columns.setdefault(key, line)

    def _steps(self, call: ast.Call) -> list[str]:
        out: list[str] = []
        for n in ast.walk(call):
            if isinstance(n, ast.Call) and n is not call:
                short = (self.a.resolve(n.func) or "").rsplit(".", 1)[-1]
                if is_estimator_like(short):
                    out.append(short)
        return out

    def _handle_for_target(self, node: ast.For | ast.AsyncFor) -> None:
        """``for tr, te in kf.split(X, y)`` / ``for i, (tr, te) in enumerate(kf.split(X))``."""
        a = self.a
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate" and it.args:
            it = it.args[0]
            tgt = node.target
            if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
                tgt = tgt.elts[1]
        else:
            tgt = node.target
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "split"):
            return
        recv = it.func.value
        splitter = None
        if isinstance(recv, ast.Name):
            splitter = a.estimator_class(recv.id, node.lineno)
        elif isinstance(recv, ast.Call):
            splitter = (a.resolve(recv.func) or "").rsplit(".", 1)[-1]
        if splitter is not None and splitter not in CV_SPLITTERS and not splitter.endswith(("Fold", "Split")):
            return
        names = _target_names(tgt)
        if len(names) != 2:
            return
        ev = SplitEvent(line=node.lineno, kind="cv_split", node=it, call=self._call_info(it),
                        inputs=[n.id for n in it.args if isinstance(n, ast.Name)], splitter=splitter)
        self._bind_side(ev, names[0], Role.TRAIN, node.lineno)
        self._bind_side(ev, names[1], Role.TEST, node.lineno)
        a.splits.append(ev)
