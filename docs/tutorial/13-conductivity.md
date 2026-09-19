# 13. Ionic conductivity

`Analysis > MD > CHGNet | VASP > Analysis > Diffusion Kinetics (D0, Ea)`

Ionic conductivity is not a menu of its own. It comes out of the
kinetics fit described in
[molecular dynamics](11-molecular-dynamics.md), under **Diffusion
Kinetics**, because computing it needs diffusion coefficients at several
temperatures — the same set the Arrhenius fit needs. This page is the
physics behind the number that fit reports.

---

## The theory

The **Nernst–Einstein** relation converts a diffusion coefficient into
a conductivity:

```
σ = (n q² / k_B T) · D
```

| | |
|---|---|
| `n` | number density of the mobile ion, per m³ |
| `q` | its charge, `z·e` |
| `D` | diffusion coefficient, m²/s |

The physical content is that both diffusion and conduction are the same
ionic motion — diffusion driven by a concentration gradient, conduction
by an electric field — and the fluctuation–dissipation theorem relates
the response to the equilibrium fluctuation.

So this follows on from [molecular dynamics](11-molecular-dynamics.md):
you need a `D` first, and preferably an activation energy so you can
quote σ at a temperature you did not simulate.

## The correlation caveat

Nernst–Einstein assumes ions move **independently**. They do not.

Measured conductivity relates to the **charge** diffusion coefficient
(the collective motion of the whole ion population), while MSD gives the
**tracer** diffusion coefficient (following individual ions). The ratio
is the Haven ratio:

```
H_R = D_tracer / D_charge
```

`H_R < 1` when motions are correlated — as in essentially every real
superionic conductor, where ions move in concerted chains rather than
one at a time. The [Van Hove distinct part](11-molecular-dynamics.md)
is what shows you this directly.

Ignoring correlation typically **underestimates** conductivity, often by
a factor of two or more.

**Quote the Nernst–Einstein value as an estimate and say so.** For the
order-of-magnitude question — is this 10⁻³ or 10⁻⁶ S/cm — it is
entirely adequate.

## For context

A solid electrolyte needs roughly **10⁻³ S/cm at room temperature** to
be practical. Reaching that conclusion from a simulated `D` at
600–1000 K means extrapolating over a long range, so the activation
energy carries most of the weight:

```
σ(300 K) = σ(1000 K) · (1000/300) · exp[-(E_a/k_B)(1/300 - 1/1000)]
```

An error of 0.1 eV in `E_a` moves the room-temperature answer by about
an order of magnitude. Get `E_a` from three or more temperatures, not
two.

## What about electronic conductivity?

QVista does not compute it, and an earlier version of this page
described a menu that no longer exists.

The reason it was withdrawn is worth stating, because the same trap
catches a lot of transport calculations. Electronic transport in the
constant relaxation-time approximation yields `σ/τ`, not `σ`: the
relaxation time depends on scattering from phonons, defects and
boundaries, and cannot be recovered from the band structure. A number
reported without a stated `τ` is not a conductivity, and a `τ` guessed
at ~10⁻¹⁴ s is an assumption doing more work than the calculation
underneath it.

For a solid electrolyte the question you actually need answering is
usually cruder, and the band gap answers it:

- gap > ~3 eV → negligible electronic conduction, good electrolyte
- small or no gap → disqualified as an electrolyte however well it
  conducts ions

That comes out of [electronic structure](07-electronic-structure.md),
where the gap is reported directly. For a **cathode** the requirement
inverts: you want both good ionic *and* good electronic conductivity,
which is why cathodes are mixed conductors, and why poor electronic
conductivity is routinely fixed with a carbon coating rather than by
changing the chemistry.

If you need a real `σ(T)` for electrons, use a dedicated Boltzmann
transport code — BoltzTraP2 or similar — on a k-mesh far denser than
the one that converged your total energy. Transport weights band
**slopes** near E_F, not band energies, and a mesh that gives a
perfectly converged energy will give a badly converged transport
integral.

---

**Next:** [Plots and export](14-plots-and-export.md).
