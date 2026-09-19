# 12. Mechanical properties

`Analysis > Mechanical Properties`

Elastic constants, and the moduli that follow from them.

---

## The theory

For small strains, stress is linear in strain — generalised Hooke's law:

```
σ_ij = Σ_kl  C_ijkl  ε_kl
```

The fourth-rank tensor `C` has 81 components, but symmetry of stress,
symmetry of strain, and the existence of a strain energy reduce it to 21
independent constants, written as a symmetric 6×6 matrix in Voigt
notation:

```
1 = xx    2 = yy    3 = zz    4 = yz    5 = xz    6 = xy
```

Equivalently, `C` is the curvature of the energy with respect to strain:

```
C_ij = (1/V₀) ∂²E / ∂ε_i ∂ε_j
```

Elastic constants are **second derivatives**, like phonons, and they are
just as sensitive to convergence.

### How it is computed

```
for each of the six independent strains ε_i:
    for a few small magnitudes (±0.5%, ±1%):
        apply the strain to the cell
        relax the ions at fixed cell shape       # ISIF = 2
        record the stress tensor

    C_ij = d σ_j / d ε_i        from a linear fit through the origin
```

The ionic relaxation at each strain is essential — it captures the
*relaxed* (rather than clamped-ion) elastic constants, which is what
experiment measures.

---

## The CHGNet route

`Analysis > Mechanical Properties > CHGNet`

QVista applies the strains, relaxes with the force field, and fits.
Minutes, locally.

---

## The VASP route

`Analysis > Mechanical Properties > VASP`

![Reading elastic constants from VASP](images/mechanical-vasp.png)

Reads a finished calculation. The usual way to produce one is
`IBRION = 6` with `ISIF = 3`, where VASP applies the distortions itself.

**Elastic constants need tighter settings than a relaxation:**

- a **denser k-mesh** — stress converges more slowly than energy
- **`ENCUT` about 1.3× the largest `ENMAX`**, because Pulay stress from
  an incomplete basis directly contaminates the stress tensor
- a **very well relaxed** reference structure

---

## What comes out

The 6×6 matrix `C_ij` in GPa, and the polycrystalline averages.

### Voigt, Reuss and Hill

Three averages of the same tensor, differing in what they assume is
uniform through a polycrystal:

| | Assumes | Gives |
|---|---|---|
| **Voigt** | uniform **strain** | upper bound |
| **Reuss** | uniform **stress** | lower bound |
| **Hill** | average of the two | the conventional quote |

```
B_V = [(C₁₁+C₂₂+C₃₃) + 2(C₁₂+C₁₃+C₂₃)] / 9
B_R = 1 / [(S₁₁+S₂₂+S₃₃) + 2(S₁₂+S₁₃+S₂₃)]        S = C⁻¹
B_H = (B_V + B_R) / 2
```

**Quote Hill.** The Voigt–Reuss gap is itself informative: a wide gap
means the crystal is strongly anisotropic and a single number is hiding
direction-dependence that may matter.

### Derived quantities

```
E  = 9BG / (3B + G)              Young's modulus
ν  = (3B - 2G) / (2(3B + G))     Poisson's ratio
```

**Pugh's ratio** `B/G` is the common ductility indicator: above ~1.75
conventionally ductile, below brittle. The physical argument is that `B`
resists volume change while `G` resists shape change, and a material
that resists volume change much more than shape change deforms
plastically rather than fracturing.

Poisson's ratio near 0.25 indicates central-force (ionic/covalent)
bonding; near 0.33 is typical of metals.

---

## Checking the result is physical

### Born stability criteria

The elastic tensor must be **positive definite** — every eigenvalue
positive. Physically: any strain must cost energy, or the crystal would
spontaneously deform.

For a cubic crystal this reduces to:

```
C₁₁ - C₁₂ > 0        resists tetragonal shear
C₁₁ + 2C₁₂ > 0       resists hydrostatic compression
C₄₄ > 0              resists shear
```

QVista checks these.

**If they fail, the structure is mechanically unstable.** Usually that
means it was not properly relaxed before the strains were applied — any
residual stress in the reference cell contaminates every constant.
Re-relax with `ISIF = 3` to tight forces and recompute.

A structure that fails Born stability *and* has imaginary
[phonon](09-phonons.md) modes is telling you the same thing twice: it is
not a minimum.

### Symmetry is a free check

Crystal symmetry constrains `C_ij` before you compute anything:

| Symmetry | Independent constants |
|---|---|
| cubic | 3 — `C₁₁`, `C₁₂`, `C₄₄` |
| hexagonal | 5 |
| tetragonal | 6 or 7 |
| orthorhombic | 9 |
| triclinic | 21 |

**If your cubic structure returns `C₁₁ ≠ C₂₂`, something is wrong** —
the calculation is not converged, or the structure is not as symmetric
as you thought. This is the cheapest available check on the entire
result and it costs nothing but looking.

---

**Next:** [Conductivity](13-conductivity.md).
