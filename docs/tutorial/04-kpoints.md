# 4. KPOINTS

Four menu entries, from two independent choices: **Gamma-centred** or
**Monkhorst-Pack**, each either sized from a spacing or typed in.

```
VASP Input > K-points > Gamma > K-spacing
VASP Input > K-points > Gamma > Customised
VASP Input > K-points > MP    > K-spacing
VASP Input > K-points > MP    > Customised
```

---

## The chemistry: why k-points exist at all

A crystal is infinite and periodic, so by Bloch's theorem every
electronic state carries a wavevector **k** in the first Brillouin zone:

```
ψ_nk(r) = e^{ik·r} · u_nk(r)        with u_nk periodic in the lattice
```

Any quantity that is a property of the whole crystal — the total energy,
the charge density, the density of states — is an **integral over the
Brillouin zone**:

```
n(r) = (1/V_BZ) ∫_BZ  Σ_occupied |ψ_nk(r)|²  dk
```

That integral has to be done numerically, as a weighted sum over a
finite mesh of k-points. **The mesh is the approximation**, and
converging it is not optional.

### The reciprocal relationship

The Brillouin zone is the reciprocal cell, so its size goes as `1/a`:

```
|b₁| = 2π/a       (for an orthogonal cell)
```

**A large real-space cell has a small Brillouin zone and needs few
k-points.** A small cell has a large zone and needs many. This is why a
2×2×2 supercell needs roughly half the mesh density of the unit cell in
each direction — and why a slab with 20 Å of vacuum needs exactly one
k-point along the vacuum direction, since there is no dispersion in a
direction with no periodicity worth sampling.

---

## Sizing from a k-spacing

`VASP Input > K-points > Gamma > K-spacing`

![Sizing a mesh from a k-spacing](images/kpoints-kspacing-gamma.png)

You give a target spacing in Å⁻¹ and QVista computes the subdivisions:

```
for each direction i:
    N_i = ceil( |b_i| / spacing )       # |b_i| = 2π/a_i
    N_i = max(N_i, 1)

report the mesh N₁ × N₂ × N₃
```

### Why this is the defensible way

Quoting "0.03 Å⁻¹ throughout" is reproducible across different
structures. Quoting "4×4×4" is not — the same grid means completely
different sampling on a 3 Å cell and a 12 Å cell, so a study that uses
4×4×4 for everything is under-converging the small cells or wasting
time on the large ones.

When you compare energies between structures of different size — a
formation energy, a defect energy, an intercalation voltage — they must
be sampled at the **same density**, not the same grid.

| Purpose | Spacing (Å⁻¹) |
|---|---|
| relaxation | 0.04 – 0.05 |
| accurate energies | 0.03 |
| DOS, transport | 0.02 or finer |
| metals | finer than the equivalent insulator |

Metals need more because the occupancy changes discontinuously at the
Fermi surface, and the integral is over the occupied region only — a
coarse mesh resolves that boundary badly.

---

## Typing a mesh directly

`VASP Input > K-points > MP > Customised`

![Entering a mesh directly](images/kpoints-customised-mp.png)

Enter the three subdivisions and an optional shift. Use this when you
are reproducing someone else's settings, or when one direction needs
special treatment — a slab, or a chain compound periodic in one
direction only.

---

## Gamma-centred or Monkhorst-Pack?

Both generate a uniform grid. The difference is whether the grid
includes the Γ point.

**Gamma-centred** always includes Γ.

**Monkhorst-Pack** with an even subdivision does not — the points are
offset by half a grid spacing.

### Use Gamma-centred for hexagonal cells

This is not a preference, it is a correctness issue. A Monkhorst-Pack
grid on a hexagonal lattice **breaks the hexagonal symmetry**: the
offset grid is not invariant under the 6-fold rotation, so the sampled
set of points does not respect the crystal's own symmetry. The result is
a subtly wrong energy that looks entirely reasonable, and a symmetry
reduction that produces more irreducible points than it should.

Gamma-centred is also required whenever Γ itself matters:

- **band gaps**, since the VBM or CBM often sits at Γ
- **phonons**, where the acoustic modes go to zero at Γ
- **any zone-centre property**

Monkhorst-Pack can converge marginally faster for simple cubic metals
with even meshes, and that is the extent of its advantage.

**When in doubt, use Gamma-centred.** It is never wrong; MP sometimes
is.

---

## Line mode, for band structures

A band structure is not computed on a mesh. It needs eigenvalues along
**lines** between high-symmetry points:

```
Γ → X → M → Γ → R
```

That file comes from `Analysis > Crystallography > High Symmetry
Pathway`, which derives the correct path from your structure's symmetry
— see [Symmetry and k-paths](06-symmetry.md).

Do not try to compute a band structure on a mesh, and do not run a
line-mode KPOINTS self-consistently.

---

## Convergence in practice

The only way to know your mesh is adequate is to test it:

```
for each mesh in increasing density:
    run a fixed-geometry SCF
    record the total energy per atom

stop when the change between successive meshes is below
your tolerance — typically 1 meV/atom
```

Converge the **energy difference** you actually care about, not the
absolute energy. Absolute total energies converge slowly; the difference
between two similar structures converges much faster, because the errors
largely cancel. If your result is a formation energy, converge that.

---

**Next:** [Optimisation with CHGNet](05-optimization.md) — relaxing
without a cluster.
