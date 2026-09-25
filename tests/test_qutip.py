"""Cross-validation against QuTiP (skipped when QuTiP is not installed)."""

import numpy as np
import pytest

import qthermo as qt

qutip = pytest.importorskip("qutip")
DIMS = [[2, 2, 2], [2, 2, 2]]


def chain():
    m = qt.models.spin_chain(3, J=0.4, master_equation="global")
    c_ops = [L for b in m.baths for L in b.c_ops]
    return m, c_ops


def test_steady_state_agrees_with_qutip():
    m, c_ops = chain()
    reference = qutip.steadystate(qutip.Qobj(m.H, dims=DIMS),
                                  [qutip.Qobj(L, dims=DIMS) for L in c_ops]).full()
    assert np.max(np.abs(qt.steady_state(m.H, m.baths) - reference)) < 1e-10


def test_dynamics_agree_with_mesolve():
    m, c_ops = chain()
    rho0 = np.zeros((8, 8), dtype=complex)
    rho0[7, 7] = 1.0
    times = np.linspace(0.0, 20.0, 101)
    reference = qutip.mesolve(qutip.Qobj(m.H, dims=DIMS), qutip.Qobj(rho0, dims=DIMS),
                              times, [qutip.Qobj(L, dims=DIMS) for L in c_ops],
                              options={"atol": 1e-12, "rtol": 1e-10}).states
    ours = qt.evolve(rho0, m.H, c_ops, duration=20.0, steps=100)["states"]
    assert max(np.max(np.abs(a.full() - b)) for a, b in zip(reference, ours)) < 1e-8


def test_qobj_inputs_are_accepted_everywhere():
    m, c_ops = chain()
    H = qutip.Qobj(m.H, dims=DIMS)
    left = qt.Bath("left", [qutip.Qobj(L, dims=DIMS) for L in m.baths[0].c_ops], 2.0)
    right = qt.Bath("right", [qutip.Qobj(L, dims=DIMS) for L in m.baths[1].c_ops], 1.0)
    r = qt.analyze(H, [left, right])
    assert np.isclose(r.current("left"), m.analyze().current("left"), rtol=1e-10)
    stats = qt.current_statistics(H, "left", baths=[left, right])
    assert stats.mean > 0
