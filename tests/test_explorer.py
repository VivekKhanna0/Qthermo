"""Interactive explorer page."""

import json
import re

import numpy as np

import qthermo as qt


def test_explorer_embeds_one_frame_per_value_for_each_master_equation(tmp_path):
    path = tmp_path / "valve.html"
    page = qt.explorer(lambda g: qt.models.two_qubit_heat_valve(g=g), np.linspace(0, 1, 5),
                       parameter="g", path=path)
    assert path.read_text() == page
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert data["kinds"] == ["global", "local"]
    assert all(len(data["frames"][k]) == 5 for k in data["kinds"])
    frame = data["frames"]["global"][3]
    reference = qt.models.two_qubit_heat_valve(g=0.75).analyze()
    assert np.isclose(frame["currents"]["hot"], reference.current("hot"))


def test_explorer_tracks_swept_temperatures():
    page = qt.explorer(lambda T: qt.models.spin_chain(2, T_left=T, master_equation="local"),
                       [1.5, 3.0], parameter="T_left", compare=False)
    data = json.loads(re.search(r"const D = (\{.*?\});\n", page, re.S).group(1))
    assert [f["bathT"]["left"] for f in data["frames"]["local"]] == [1.5, 3.0]
