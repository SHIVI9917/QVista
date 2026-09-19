# 8. Bonding and charge

Where the electrons are, and what is bonded to what.

---

## Bader Analysis

`Analysis > Bonding Tools > Bader Analysis`

![The Bader dialog](images/bader.png)

### The theory

Bader's **Quantum Theory of Atoms in Molecules** partitions space into
atomic basins using only the charge density — no reference to orbitals,
basis sets or arbitrary radii.

The boundary between two atoms is the **zero-flux surface**: the surface
on which the density gradient has no component along the normal.

```
∇n(r) · n̂(r) = 0        for r on the surface
```

Equivalently: follow the gradient of the density uphill from any point,
and you arrive at a nucleus. The basin of an atom is the set of all
points whose gradient path ends at that nucleus. The dividing surfaces
pass through the **bond critical points**, the saddle points of the
density.

The charge is then just the integral over the basin:

```
Q_A = Z_A - ∫_basin(A) n(r) dr
```

This is the most defensible atomic-charge definition available, because
the density is a genuine quantum-mechanical observable and the
partitioning uses only its topology.

### What you must have run

| File | |
|---|---|
| `CHGCAR` | valence pseudo-density |
| `AECCAR0` | core density |
| `AECCAR2` | all-electron valence density |

**`CHGCAR` alone gives wrong charges.** VASP's PAW pseudises the density
near each nucleus, and the sharp maxima at the nuclei are exactly what
the gradient-following algorithm keys on. Without the AECCARs there is
nothing for the basins to converge to.

Set `LAECHG = .TRUE.` **in the run that produces them.** They cannot be
recovered afterwards.

```
n_total = CHGCAR + AECCAR0 + AECCAR2      # reconstructed all-electron
assign each grid point to a nucleus by following ∇n uphill
Q_A = Z_A - Σ_{points in basin A} n · dV
```

QVista uses the Henkelman group's `bader` executable and tells you where
to get it if it is missing.

### Reading the numbers honestly

Bader charges are **systematically smaller** than formal oxidation
states. In an oxide, "O²⁻" comes back around −1.2 to −1.5.

That is not an error. A formal oxidation state is a bookkeeping
convention that assigns every shared electron entirely to the more
electronegative atom. Bader partitions real space, and no real bond is
fully ionic.

**Compare Bader charges between structures, never against formal
states.** The useful quantity is the *change*: how much does the Co
charge change when you remove Li? If it changes by ~1, the metal is
being oxidised. If it barely changes while O changes instead, you have
oxygen redox — a completely different degradation story.

---

## DDEC6

`Analysis > Bonding Tools > DDEC6`

![The DDEC6 dialog](images/ddec6.png)

A different partitioning of the same density. Where Bader draws hard
boundaries in space, DDEC6 fits **spherically averaged atomic densities**
that simultaneously:

- resemble reference free-atom densities (chemical sensibility)
- reproduce the electrostatic potential outside the material

That second condition is why DDEC6 charges are used to parameterise
classical force fields: charges that reproduce the electrostatic
potential give the right long-range interactions in an MD simulation,
which Bader charges do not aim to do.

DDEC6 also gives **bond orders**, which Bader does not.

Needs the Chargemol program and its reference density files.

**Never compare a Bader charge to a DDEC6 charge.** They are different
definitions and disagree substantially. Pick one and use it
consistently throughout a study.

---

## COHP

`Analysis > Bonding Tools > COHP`

![The COHP dialog](images/cohp.png)

### The theory

Crystal Orbital Hamilton Population resolves the DOS into **bonding**
and **antibonding** contributions for a specific atom pair:

```
COHP_AB(E) = Σ_nk  Re[ c*_A,nk · c_B,nk · H_AB ] · δ(E - ε_nk)
```

where `c` are coefficients of a local basis and `H_AB` the Hamiltonian
matrix element between orbitals on A and B.

The sign is what matters. Plotted conventionally as **−COHP**:

- **positive** → bonding: occupying this state strengthens the A–B bond
- **negative** → antibonding: occupying it weakens the bond

The integral to the Fermi level, **ICOHP**, is a bond-strength measure
in eV.

### What this tells you that a DOS cannot

A DOS says states exist at an energy. COHP says whether filling them
*helps or hurts* a particular bond. Those are different questions, and
only the second one is about bonding.

**The diagnostic that matters in battery cathodes: antibonding states
below E_F.**

If the metal–oxygen bond has occupied antibonding character, that bond
is already destabilised in the ground state. On charging, as you remove
electrons, you may empty those antibonding states — which *strengthens*
M–O — or you may start removing bonding O 2p electrons, which weakens
the framework and leads to oxygen release. Staring at a total DOS
reveals none of this.

Requires **LOBSTER**, which projects the plane-wave wavefunctions onto a
local basis. That needs `LWAVE = .TRUE.` from the VASP run — set it
beforehand.

---

## Electronegativity tools

`Analysis > Electronegativity > ...`

Four entries that work together on the structure in the viewport.

### Select Atoms

Click atoms; the panel lists them with their electronegativities and the
differences between pairs.

Pauling electronegativity difference is the crudest useful predictor of
bond character:

```
Δχ = |χ_A - χ_B|

Δχ > 1.7    conventionally "ionic"
Δχ < 1.7    conventionally "covalent"
```

Pauling's original definition came from bond energies:

```
Δχ = 0.102 √(E_AB - √(E_AA · E_BB))     in kJ/mol
```

— the extra stability of a heteronuclear bond over the geometric mean
of the homonuclear ones, attributed to ionic resonance.

It is a rule of thumb from the 1930s. The reason it is worth having in
the same program as a Bader charge and a COHP curve is that you can see
how far the rule of thumb is from what the density actually does. For
Li–O, Δχ = 2.46 says "ionic", and both the Bader charge and the DOS
agree. For Co–O, Δχ = 1.56 says "covalent", and the COHP shows real
orbital mixing. The tools corroborate each other.

### Polyhedral Environment

Finds the coordination polyhedron of a selected centre — neighbours,
coordination number, and the polyhedron's distortion.

```
neighbours = atoms within cutoff of the centre
CN         = count(neighbours)
d_mean     = mean bond length
distortion = (1/N) Σ ((d_i - d_mean) / d_mean)²
```

The distortion index is the useful output. A Jahn–Teller-active ion —
Mn³⁺ (d⁴) or Cu²⁺ (d⁹) — gives an elongated octahedron with four short
and two long bonds, and that shows up as a large distortion index before
you compute any electronic structure.

### Shift Map

Maps electronegativity differences across the cell, so polar bonds are
visible at a glance instead of pair by pair.

### Clear Selection

Clears what the other three are working on. Worth knowing about: a stale
selection is the usual reason one of them appears to do nothing.

---

**Next:** [Phonons](09-phonons.md) — is the structure even stable?
