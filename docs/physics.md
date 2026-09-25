# What qthermo computes, exactly

This page states every definition and convention behind the numbers the
package reports, so a result can be checked against a paper, or a paper
against a result, without reading the source. Units: `ħ = k_B = 1`.

## Signs

- `Q > 0`, `J_k > 0`: energy flows **into** the system (from bath `k`).
- `W > 0`: work is done **on** the system.
- `dU = Q + W`.
- `qubit_hamiltonian(ω) = −(ω/2) σ_z`: |0⟩ is the ground state and |1⟩ the
  excited state, at `+ω/2`.
- `sigma_minus = |0⟩⟨1|` lowers the energy.

## Master equation

    dρ/dt = −i[H, ρ] + Σ_L ( L ρ L† − ½{L†L, ρ} )

Superoperators use column stacking, `vec(AρB) = (Bᵀ ⊗ A) vec(ρ)`, with
`vec(ρ) = ρ.reshape(-1, order="F")`.

## Heat and work along a trajectory of states (strokes)

On a stored grid `t_k`, each step contributes

    W_k = Tr[(H_{k+1} − H_k)(ρ_k + ρ_{k+1})/2]
    Q_k = Tr[(H_k + H_{k+1})/2 (ρ_{k+1} − ρ_k)]

This is the midpoint form of Alicki's split. `Q_k + W_k = ΔU_k` holds
*exactly*, so `first_law_residual` measures solver error, not quadrature error.

With several `Bath` objects on one stroke, bath `k`'s share is
`∫ Tr[H(t) D_k(t)[ρ(t)]] dt` by the trapezoid rule on the same grid. The shares
are then rescaled to sum to the exact total. The rescaling is `O(dt²)` and
keeps the first law exact per stroke.

Entropy production of a stroke:

- one bath: `σ = ΔS − Q/T`;
- several baths: `σ = ΔS − Σ_k Q_k/T_k`.

## Baths

### Global (Davies) baths — `davies_bath(H, A, T, ...)`

`A` is the Hermitian system operator the bath couples through. It is split into
Bohr components of the full `H`:

    A(ω) = Σ_{ε'−ε=ω} P(ε) A P(ε')         (lowers the energy by ω)

Degenerate levels and degenerate Bohr frequencies are grouped within
`tol = 1e-9 × max(spectral width, 1)`. Each group becomes one jump operator
`L_ω = √γ(ω) A(ω)` with

    γ(ω)  = J(ω) (1 + n(ω))     ω > 0
    γ(−ω) = J(ω) n(ω)           n(ω) = 1/(e^{ω/T} − 1)

so `γ(−ω)/γ(ω) = e^{−ω/T}` exactly, and `e^{−H/T}/Z` is a fixed point.

`J(ω)` is the **zero-temperature emission rate** at Bohr frequency ω, not a
spectral density in any paper's normalisation:

| spectrum | `J(ω)` | ω → 0 channel |
|---|---|---|
| `flat_spectrum(γ)` (default) | `γ` | rate undefined → off (warns if it connects distinct degenerate states) |
| `ohmic_spectrum(γ, cutoff, reference)` | `γ (ω/reference) e^{−ω/cutoff}` | `γ T / reference` |

With the flat spectrum on a single qubit coupled through σ_x, this is exactly
`thermal_bath(γ, ω, T)`. The Lamb shift is neglected.

Operators are stored sparse in the eigenbasis of `H`; `Bath.c_ops` is the dense
computational-basis view, built on demand.

### Local baths — `local_bath(h_site, A_site, T, site, dims)`

The Davies construction on the isolated site's Hamiltonian, embedded in the
full space. This is the local master equation. It is valid when inter-site
couplings are small compared with the bath rates. Its fixed point is not the
Gibbs state of the coupled `H`.

### Time-dependent baths — `instantaneous_bath(H_of_t, A, T, ...)`

The Davies bath of the instantaneous `H(t)` at each time. This is the adiabatic
Markovian master equation (Albash et al., NJP 14, 123016 (2012)), valid for
driving slow compared with the bath correlation time.

## Steady states — `steady_state`, `analyze`

`L ρ = 0` is solved with one row replaced by `Tr ρ = 1`. That matrix is
singular exactly when the steady state is not unique. Its LAPACK
reciprocal-condition estimate (or the sparse LU pivots) is the uniqueness
test, with threshold `1e-13`. A non-unique steady state raises, unless
`initial_state` is given; then the result is the spectral projection of
`initial_state` onto the kernel of `L`.

When every bath is a Davies bath of the same `H`, the solve runs in the
eigenbasis of `H`, where the jump operators are sparse.

For superoperators larger than 16384 (above 128 levels), the bordered system
is first reordered by reverse Cuthill–McKee. It is then solved with GMRES
(relative tolerance `1e-12`, restart 200), preconditioned by an incomplete LU
with drop tolerance `1e-4`. The residual is checked, and the solver falls back
to sparse LU if GMRES fails.

### Heat currents and entropy production

    J_k = Tr[E D_k(ρ_ss)]
    σ̇  = −Σ_k J_k / T_k            (steady state: dS/dt = 0)

`E` is the **energy operator**:

- `analyze(H, baths)` uses `E = H`. Then `Σ_k J_k = 0` exactly for a static
  `H`.
- `Model.analyze()` uses `E = H0` (the non-interacting Hamiltonian) when the
  baths are local. This is the thermodynamically consistent choice for the
  local master equation (De Chiara et al., NJP 20, 113024 (2018)). The currents
  then do not sum to zero; `work_rate = −Σ_k J_k` is the power needed to keep
  the coupling switched on ("boundary work").
- Rotating-frame models (the maser) use the bare `H0`. There
  `power_output = Σ_k J_k` is the power delivered to the drive.

Figures of merit:

- refrigerator `cop(cold, source) = J_cold / J_source`;
- absorption Carnot bound `(1 − T_r/T_h)/(T_r/T_c − 1)`;
- engine `efficiency(hot) = power_output / J_hot`.

`relaxation_time(H, baths)` is `1/gap` of the Liouvillian.
`populations_only=True` uses the gap of the classical rate matrix
`W_ij = Σ_L |⟨i|L|j⟩|²` between energy eigenstates.

## Site-resolved flows — `heat_flow_map`

For `H = Σ_i h_i + Σ_b V_b`:

    J_{k→i} = Tr[h_i D_k(ρ)]               bath k into site i
    J_{b→i} = i Tr(ρ [V_b, h_i])           interaction term b into site i
    d⟨h_i⟩/dt = Σ_k J_{k→i} + Σ_b J_{b→i}   (= 0 in a steady state)

The **bond current** of a two-site term is `(J_{b→j} − J_{b→i})/2`, from `i`
to `j`. Energy stored in the bond cancels out of this.

In a global-ME steady state `[H, ρ] = 0`, so every `J_{b→i}` vanishes
identically and bath heat enters the interaction energy. This is flagged
(`extras["secular_blind"]`), not hidden.

The **virtual temperature** of a site is
`T* = (e₁ − e₀) / ln(p₀/p₁)` from the two lowest eigenstates of `h_i` in the
reduced state (Brunner et al., PRE 85, 051117 (2012)). It is negative for an
inversion, and `inf` for equal populations.

Correlation measures:

- **Mutual information** `S_a + S_b − S_ab`.
- **Negativity** `(‖ρ^{T_b}‖₁ − 1)/2`.
- **Concurrence** Wootters' formula, for qubit pairs.

## Fluctuations — `current_statistics`, `scaled_cgf`

A counted current assigns each jump operator `L_j` a weight `ν_j`:

- `"energy"`: the energy the jump deposits, `[E, L_j] = ν_j L_j`. It raises
  if `L_j` is not an eigenoperator of `E`.
- `"quanta"`: `±1`.
- a number or list: the weights given.

With `𝒥(ρ) = Σ_j ν_j L_j ρ L_j†`:

    J = Tr[𝒥 ρ_ss]
    D = Σ_j ν_j² Tr[L_j ρ_ss L_j†] − 2 Tr[𝒥 L^D 𝒥 ρ_ss]   = lim Var[N_t]/t
    K = Σ_j Tr[L_j ρ_ss L_j†]                               (dynamical activity)

`L^D` is the Drazin inverse, applied through a bordered linear system (Landi et
al., PRX Quantum 5, 020201 (2024)). `D` is the full variance rate, not half of
it.

- **TUR ratio** `(D/J²) σ̇`: at least 2 for classical Markov jump processes.
- **KUR ratio** `(D/J²) K`: at least 1 for classical Markov jump processes.
- `scaled_cgf(s)`: the eigenvalue with largest real part of
  `L + Σ_j (e^{s ν_j} − 1) L_j ⊗ L_j*`.

## Strong coupling — `reaction_coordinate_model`

    H_ext = H_S + Ω a†a + λ S (a + a†) + (λ²/Ω) S²

The last term is the counterterm, on by default. The residual bath is a Davies
bath on `H_ext` coupled through `a + a†`, with `ohmic_spectrum(κ,
reference=Ω)`, so the RC decays at rate κ at its own frequency. Parameters are
stated in this rate convention rather than through a spectral-density formula,
whose prefactors differ between papers.

`mean_force_state` is `Tr_RC e^{−H_ext/T}/Z`. `ultrastrong_limit_state` is
`Σ_n P_n e^{−P_n H_S P_n/T} P_n / Z`, with `P_n` the eigenprojectors of `S`
(Cresser & Anders 2021).

## Erasure — `landauer_erasure`

- **Qubit:** `H(t) = qubit_hamiltonian(ω(t))`, with `ω` ramped from `ω_min`
  to `ω_max` (default `12T`).
- **Bath:** Ohmic, through σ_x, with `reference = T`, following `H(t)`.
- **Initial state:** maximally mixed.

Reported quantities:

- `heat_to_bath = −Q`.
- `landauer = T (S_i − S_f)`.
- `excess_heat = heat_to_bath − landauer = T σ ≥ 0`.

Slow-driving friction:

    ζ(ω) = β Var(∂_ω H) / Γ(ω),   Var = p(1−p),   Γ = γ (ω/T) coth(ω/2T)
    excess ≈ (1/τ) ∫₀¹ ζ(ω(s)) ω'(s)² ds  ≥  L²/τ,   L = ∫ √ζ dω

`geodesic_schedule` runs at constant `√ζ ω̇`, which attains `L²/τ`.

## Optimal protocols — `friction`, `optimal_schedule`

For `H(λ)` with a thermalising generator `𝓛_λ` (fixed point `π_λ`, the Gibbs
state at `T`), slow driving leaves the state lagging by `δρ = 𝓛⁺(∂_λπ) λ̇`, and

    W − ΔF ≈ ∫ g(λ) λ̇² dt,     g(λ) = Tr[∂_λH · 𝓛⁺(∂_λπ)]

where `𝓛⁺` is the Drazin inverse, applied through the bordered solve. The
derivatives are central differences with `h = 1e-5`. The metric includes
coherent contributions when `[H, ∂_λH] ≠ 0`.

Along a path `λ(s)`, the constant-speed schedule `√g |dλ/ds| ṡ = const`
minimises the excess, with value `L²/τ`. `excess_work` simulates the stroke
with the dissipator rebuilt at every time (adiabatic master equation) and
reports `T σ`.

## Otto cycles with interacting media — `ideal_otto`, `otto_cycle`

`ideal_otto` gives the quasi-static cycle:

- **Isochores:** full thermalisation.
- **Driven strokes:** quantum adiabatic.
- **Heat:** `Q_h = Σ_n E^h_n (p^h_n − p^c_n)`, where `n` pairs a level of
  `H_hot` with its **adiabatic continuation** in `H_cold`. The pairing is found
  by overlap-matching eigenvectors along the drive path. Pass
  `follow_crossings=False` for energy-rank pairing.

`otto_cycle` is the finite-time version:

- **Baths:** one Davies bath per coupling operator.
- **Drive:** a linear (or given) interpolation between `H_c` and `H_h`.
- **Refusals:** it refuses baths that cannot thermalise, i.e. a non-unique
  steady state.
- **Warnings:** it warns if `τ_iso < 5 ×` the population relaxation time.

## Periodically driven machines — `floquet_analyze`

The Floquet modes `|u_a(t)⟩` come from the one-period propagator. It is built
as a product of midpoint exponentials: `n_time × substeps` slices. The
quasienergies `ε_a = −arg(λ_a)/T_d` lie in `(−Ω/2, Ω/2]`.

For each bath, `⟨u_a(t)|A|u_b(t)⟩` is sampled on `n_time` points and
Fourier-transformed: `A_ab(q)` is the coefficient of `e^{+iqΩt}`. The
transition `b → a` in sideband `q` hands the bath the energy
`w = ε_b − ε_a − qΩ`, at rate `γ(w)|A_ab(q)|²` with the same `γ` as the
Davies construction.

The Floquet populations follow the Pauli rate equation built from the `a ≠ b`
terms. The currents are

    J_k = −Σ w γ_k(w) |A_ab(q)|² p_b

summed over all terms, including `a = b`, `q ≠ 0`. The power delivered to the
drive is `P = Σ_k J_k`. A warning is issued when distinct transition
frequencies lie closer than `1e-3 ×` the largest rate.

## Linear response — `response`

`G_kl = ∂J_k/∂T_l` is computed by central differences with step `h·T_l`
(`h = 1e-4`), each point a full steady-state solve. The Onsager matrix is
`L_kl = T_l² G_kl`, i.e. the response to the affinities `x_l = 1/T − 1/T_l`.

- `reciprocity_residual`: `max|L − Lᵀ| / max|L|`, meaningful at equilibrium.
- `conservation_residual`: `max_l |Σ_k G_kl| / max|G|`.
- `coupling(a, b)`: `L_ab / √(L_aa L_bb)`.
- `amplification(control, output)`: `G_oc / G_cc`, the change in the output
  current per change in the control current when only `T_control` is varied.

## Transients — `transient`

The master equation is integrated from `ρ₀` with `evolve` (DOP853,
`rtol = 1e-9`). At every output time the currents `J_k(t) = Tr[E D_k(ρ(t))]`
are recorded. The cumulative heats are trapezoid integrals of `J_k(t)`, so
`energy_balance_residual` measures quadrature accuracy.
`product_thermal_state` is `⊗_i e^{−h_i/T_i}/Z_i`.

## Exact counting statistics of cycles — `cycle_counting`

Each jump operator `L_j` of a counted stroke gets a weight `ν_j`: `+1` if it
raises the energy of the stroke's Hamiltonian, `−1` if it lowers it (times a
number, if one is given). The tilted generator is

    𝓛(χ) = 𝓛 + Σ_j (e^{iχν_j} − 1) L_j ⊗ L_j*

Each stroke propagator is `exp(𝓛(χ) τ)` for constant strokes, and a
midpoint-rule product over `substeps` slices otherwise.
`G(χ) = Tr[P_N(χ)…P_1(χ) ρ_start]`, and `P(n)` is its discrete Fourier
transform on `4 n_max` points. By default `ρ_start` is the limit-cycle state.
The long-run statistics per cycle come from `θ(s) = ln Λ(s)`, the dominant
eigenvalue of the tilted one-cycle propagator, with `θ'(0)` the mean and
`θ''(0)` the variance, by finite differences with `h = 1e-3`.

## Operation modes — `classify`

With `W` the work done on the machine and `Q_h`, `Q_c` the heats in:

| mode | signs |
|---|---|
| engine | `W<0, Q_h>0, Q_c<0` |
| refrigerator | `W>0, Q_c>0, Q_h<0` |
| accelerator | `W>0, Q_h>0, Q_c<0` |
| heater | `W>0, Q_h≤0, Q_c≤0` |

Magnitudes below `1e-12 ×` the largest count as zero. Any other pattern is
`forbidden`, i.e. it violates the second law.

For a `SteadyState`, `W = work_rate`. That is non-zero only when the currents
are measured with an energy operator that makes work explicit (local baths
with `H0`, rotating frames).

## Batteries — `qthermo.batteries`

- **incoherent ergotropy:** the ergotropy of `ρ` dephased in the eigenbasis of
  `H`, keeping coherences inside degenerate eigenspaces.
- **coherent ergotropy:** total minus incoherent.
- **locked ergotropy:** `W(ρ) − Σ_i W(ρ_i)` for `H = Σ_i h_i`.
- **asymptotic ergotropy:** `U(ρ) − U(G_β)`, where `S(G_β) = S(ρ)` and `β`
  is found by bracketing.
- **Dicke battery:** `H = ω(J_z + N/2) + ω a†a + g(J_+ + J_−)(a + a†)` in the
  symmetric subspace, with the cavity truncated at `photons + N + 6`.
  Charging power is `max_t E(t)/t` over the window `1.2π/g`.
  `collective_advantage` fits the exponent on the three largest N.

## Multi-qubit views of cycles — `site_dynamics`

Evaluated at every stored state of every stroke:

- per site: `⟨h_i⟩`, the virtual temperature and (for qubits) the Bloch vector
  `⟨σ_{x,y,z}⟩`;
- per pair: mutual information and concurrence;
- the total correlation `Σ_i S_i − S`.

## Model audit — `audit`

The checks run are:

- **uniqueness:** the steady-state solve.
- **relaxation time:** `relaxation_time(populations_only=True)`; the full
  Liouvillian gap is also computed up to 16 levels.
- **second law:** `σ̇ ≥ 0`.
- **direction:** heat must not flow from the colder bath to the hotter one
  without work.
- **detailed balance vs declared T:** for each pair of jump operators that are
  adjoint up to scale, with energy change `ω`, the implied
  `T = ω / ln(|L_down|²/|L_up|²)`. The finding reports the median.
- **local vs global:** for rebuildable models, the largest relative current
  difference and any sign flips.
- **internal currents:** whether they vanish because the steady state is
  diagonal in H.
- **uncertainty relations:** TUR and KUR ratios of each bath's heat current,
  up to 32 levels.

## Trajectories — `unravel`

Monte Carlo wave-function unravelling, all trajectories propagated together.
Jump times are resolved to one time step, which biases the statistics at
`O(rate × dt)`; `cycle_counting` gives the exact answer to compare against.
Heats within `1e-9 ×` scale of zero are set to exactly zero, so that
`q <= 0` counts trajectories whose jumps cancel.

- **Heat:** the energy change `⟨ψ|H|ψ⟩` across each jump.
- **Work:** the first-law remainder on each trajectory.

The ensemble mean reproduces the master-equation heat within statistical error
(tested).
