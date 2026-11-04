from __future__ import annotations

from pathlib import Path

from leakray import analyze_file


def _lk012(tmp_path: Path, code: str) -> list:
    p = tmp_path / "case.py"
    p.write_text(code, encoding="utf-8")
    return [f for f in analyze_file(p).findings if f.rule == "LK012"]


def test_constructed_splitter_without_loop_is_flagged(tmp_path: Path) -> None:
    code = (
        "import pandas as pd\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\n"
        "df = pd.read_csv('x.csv')\nids = df['patient_id']\nskf = StratifiedKFold(5, shuffle=True)\n"
        "cross_val_score(None, df, df['y'], cv=skf)\n"
    )
    assert len(_lk012(tmp_path, code)) == 1


def test_split_loop_is_reported_once(tmp_path: Path) -> None:
    code = (
        "import pandas as pd\nfrom sklearn.model_selection import KFold\n"
        "df = pd.read_csv('x.csv')\nids = df['user_id']\nkf = KFold(5, shuffle=True)\n"
        "for tr, te in kf.split(df):\n    pass\n"
    )
    assert len(_lk012(tmp_path, code)) == 1
