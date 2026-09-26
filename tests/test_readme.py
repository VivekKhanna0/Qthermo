"""Every Python block in the README must run as written."""

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_code_blocks_execute():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert blocks, "no python blocks found"
    code = "import qthermo as qt\n" + "\n".join(blocks)
    # keep CI fast: the trajectory example is statistical, not a check
    code = code.replace("trajectories=6000", "trajectories=100")
    exec(compile(code, "README.md", "exec"), {})


def test_readme_benchmark_count_is_current():
    from qthermo.benchmarks import REGISTRY, _load_all
    _load_all()
    stated = re.search(r"# (\d+) checks against published results", README.read_text())
    assert stated, "README should state how many benchmark checks there are"
    assert int(stated.group(1)) == len(REGISTRY), (
        f"README says {stated.group(1)} checks; the registry has {len(REGISTRY)}")
