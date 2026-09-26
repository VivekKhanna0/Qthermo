"""Every Python block in the README must run as written."""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "guide.md"


@pytest.mark.parametrize("page", [README, GUIDE], ids=["README", "guide"])
def test_readme_code_blocks_execute(page, tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)          # files the examples write land here
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.S)
    assert blocks, "no python blocks found"
    code = "import qthermo as qt\n" + "\n".join(blocks)
    # keep CI fast: the trajectory example is statistical, not a check
    code = code.replace("trajectories=6000", "trajectories=100")
    exec(compile(code, page.name, "exec"), {})


def test_readme_benchmark_count_is_current():
    from qthermo.benchmarks import REGISTRY, _load_all
    _load_all()
    stated = re.search(r"# (\d+) checks against published results", README.read_text())
    assert stated, "README should state how many benchmark checks there are"
    assert int(stated.group(1)) == len(REGISTRY), (
        f"README says {stated.group(1)} checks; the registry has {len(REGISTRY)}")
