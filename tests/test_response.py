"""Linear response: Onsager reciprocity, tight coupling, transistor gain."""

import numpy as np

import qthermo as qt


def fridge(master_equation):
    return lambda T: qt.models.absorption_refrigerator(
        T_c=T["cold"], T_h=T["hot"], T_r=T["room"], master_equation=master_equation)


def transistor(T):
    return qt.models.thermal_transistor(T_L=T["L"], T_M=T["M"], T_R=T["R"])


EQ3 = {"cold": 1.0, "hot": 1.0, "room": 1.0}


def test_onsager_reciprocity_and_second_law_for_global_baths():
    for build, T in ((fridge("global"), EQ3), (transistor, {"L": 0.5, "M": 0.5, "R": 0.5})):
        r = qt.response(build, T)
        assert r.at_equilibrium
        assert r.reciprocity_residual < 1e-6
        assert r.conservation_residual < 1e-8
        assert r.is_positive_semidefinite()


def test_local_absorption_fridge_is_tightly_coupled():
    r = qt.response(fridge("local"), EQ3)
    assert np.isclose(abs(r.coupling("cold", "hot")), 1.0, atol=1e-6)
    assert np.isclose(abs(r.coupling("hot", "room")), 1.0, atol=1e-6)


def test_transistor_gain_exceeds_one_and_obeys_the_sum_rule():
    r = qt.response(transistor, {"L": 1.0, "M": 0.1, "R": 0.2})
    to_R, to_L = r.amplification("M", "R"), r.amplification("M", "L")
    assert abs(to_R) > 3.0
    assert np.isclose(to_R + to_L, -1.0, atol=1e-6)     # energy conservation


def test_response_rejects_unknown_bath():
    import pytest
    with pytest.raises(qt.QThermoError):
        qt.response(fridge("global"), {"cold": 1.0, "hot": 1.0, "nope": 1.0})
