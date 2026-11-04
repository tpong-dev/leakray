"""Unit tests for the benchmark metric code (synthetic input only, never real benchmark numbers)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUN_PY = Path(__file__).resolve().parents[1] / "benchmark" / "run.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("benchmark_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_run"] = module
    spec.loader.exec_module(module)
    return module


def case(run, name, expected, predicted):
    return run.CaseResult(name, set(expected), set(predicted))


def test_parse_expect(run):
    assert run.parse_expect("# expect: clean") == set()
    assert run.parse_expect("# expect: LK001") == {"LK001"}
    assert run.parse_expect("#expect: lk001, LK009 ") == {"LK001", "LK009"}
    assert run.parse_expect("import pandas as pd") is None


def test_per_rule_counts_and_scores(run):
    results = [
        case(run, "a.py", ["LK001"], ["LK001"]),  # TP LK001
        case(run, "b.py", ["LK001"], []),  # FN LK001
        case(run, "c.py", [], ["LK001"]),  # FP LK001
        case(run, "d.py", ["LK002", "LK003"], ["LK002"]),  # TP LK002, FN LK003
        case(run, "e.py", [], []),  # TN
    ]
    m = run.compute_metrics(results)
    lk1 = m["per_rule"]["LK001"]
    assert (lk1["tp"], lk1["fp"], lk1["fn"]) == (1, 1, 1)
    assert lk1["precision"] == pytest.approx(0.5)
    assert lk1["recall"] == pytest.approx(0.5)
    assert lk1["f1"] == pytest.approx(0.5)
    assert lk1["fp_files"] == ["c.py"]
    assert lk1["fn_files"] == ["b.py"]
    assert m["per_rule"]["LK002"]["precision"] == pytest.approx(1.0)
    lk3 = m["per_rule"]["LK003"]
    assert (lk3["tp"], lk3["fp"], lk3["fn"]) == (0, 0, 1)
    assert lk3["precision"] == 0.0
    assert lk3["f1"] == 0.0

    micro = m["micro"]
    assert (micro["tp"], micro["fp"], micro["fn"]) == (2, 1, 2)
    assert micro["precision"] == pytest.approx(2 / 3)
    assert micro["recall"] == pytest.approx(0.5)


def test_file_level_any_finding(run):
    results = [
        case(run, "a.py", ["LK001"], ["LK004"]),  # wrong rule but still detected at file level
        case(run, "b.py", ["LK001"], []),
        case(run, "c.py", [], ["LK015"]),
        case(run, "d.py", [], []),
    ]
    fl = run.compute_metrics(results)["file_level"]
    assert (fl["tp"], fl["fp"], fl["fn"], fl["tn"]) == (1, 1, 1, 1)
    assert fl["precision"] == pytest.approx(0.5)
    assert fl["recall"] == pytest.approx(0.5)


def test_empty_input_has_no_division_errors(run):
    m = run.compute_metrics([])
    assert m["per_rule"] == {}
    assert m["micro"]["precision"] == 0.0
    assert m["file_level"]["recall"] == 0.0


def test_markdown_table_lists_rules(run):
    m = run.compute_metrics([case(run, "a.py", ["LK001"], ["LK001"])])
    table = run.markdown_table(m)
    assert "| LK001 | 1 | 0 | 0 |" in table
    assert "micro" in table


def test_read_label_from_notebook(run, tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
             "source": ["# expect: LK006\n", "import pandas as pd"]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert run.read_label(path) == {"LK006"}
    py = tmp_path / "y.py"
    py.write_text("# expect: clean\nx = 1\n", encoding="utf-8")
    assert run.read_label(py) == set()
