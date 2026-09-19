"""
Which pairs of elements are drawn as bonded, and out to what distance.

Bonds in a crystal viewer are a drawing convention, not a measurement.
The usual convention -- join two atoms whenever they sit closer than the
sum of their covalent radii, times a little slack -- is a good default
and wrong often enough to be worth overriding.

An intercalation cathode is the case that forces the issue. In
LiCoO2 the lithium sits between the CoO2 slabs at a distance that clears
any covalent criterion, so the default draws Li-O bonds and the layered
structure disappears into a solid block. The Li is not covalently bonded
to anything; it is an intercalated cation, and the picture that shows
that is the one with the Li-O sticks taken away. Nothing about the
geometry can be inferred to make that call, so it is offered as a
choice: per element pair, drawn or not, and out to what distance.

A rule is stored only once it differs from the default. That keeps the
common case free of state, and it means a pair left alone still follows
its covalent radii if the structure changes underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def covalent_radius(symbol: str) -> float:
    """
    The tabulated covalent radius, in Angstroms.

    Imported on the first call rather than at the top of the file.
    :mod:`qvista.visualization.settings` holds a :class:`BondRules` and
    so imports this module, while :mod:`qvista.visualization.radii`
    imports ``settings`` for the representation enum -- so importing it
    up here closes a cycle and neither module finishes loading. The call
    is a dictionary lookup once Python has cached the module, and it
    happens a few times per redraw, not per atom pair.
    """

    from qvista.visualization.radii import covalent_radius as radius

    return radius(symbol)


#: How much longer than the sum of the covalent radii still counts as a
#: bond. Radii are tabulated for the ideal two-centre case and real ones
#: stretch, so a criterion without slack misses bonds that are plainly
#: there.
DEFAULT_SCALE = 1.15

#: A cutoff of zero means "whatever the covalent radii say", so that a
#: rule can switch a pair off without also freezing its distance.
FROM_RADII = 0.0

#: How far a cutoff may be dragged, in Angstroms. Past about 6 A every
#: pair in an ordinary cell is within range and the picture is a solid
#: mass of cylinders.
MIN_CUTOFF = 0.0
MAX_CUTOFF = 6.0


def pair_key(a: str, b: str) -> tuple[str, str]:
    """
    The canonical key for an element pair.

    A bond has no direction, so Li-O and O-Li have to land on the same
    rule. Sorting is what makes that true regardless of which atom the
    builder happened to reach first.
    """

    return (a, b) if a <= b else (b, a)


def element_pairs(elements) -> list[tuple[str, str]]:
    """
    Every pair that could be drawn among these elements, homonuclear
    pairs included.

    Three elements give six pairs, not three: Co-Co and O-O are real
    contacts in a dense oxide, and O-O in particular is the one a user
    wants to switch off after widening some other cutoff.
    """

    unique = sorted(set(elements))

    return [
        (unique[i], unique[j])
        for i in range(len(unique))
        for j in range(i, len(unique))
    ]


@dataclass(slots=True)
class BondRule:
    """One element pair's departure from the default."""

    #: Whether this pair is drawn at all.
    enabled: bool = True

    #: Longest separation still drawn as a bond, in Angstroms.
    #: :data:`FROM_RADII` defers to the covalent radii.
    cutoff: float = FROM_RADII


@dataclass(slots=True)
class BondRules:
    """
    The bonding convention in force.

    Empty means the plain covalent criterion for everything, which is
    what a freshly opened structure gets.
    """

    scale: float = DEFAULT_SCALE

    #: Only the pairs that have been changed. See the module docstring.
    rules: dict[tuple[str, str], BondRule] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def default_cutoff(self, a: str, b: str) -> float:
        """What the covalent radii alone would allow, in Angstroms."""

        return self.scale * (covalent_radius(a) + covalent_radius(b))

    def rule_for(self, a: str, b: str) -> BondRule | None:
        """The stored rule for this pair, or ``None`` if it is untouched."""

        return self.rules.get(pair_key(a, b))

    def is_enabled(self, a: str, b: str) -> bool:
        """Whether this pair is drawn."""

        rule = self.rule_for(a, b)

        return True if rule is None else rule.enabled

    def cutoff(self, a: str, b: str) -> float:
        """The distance out to which this pair is drawn, in Angstroms."""

        rule = self.rule_for(a, b)

        if rule is None or rule.cutoff == FROM_RADII:
            return self.default_cutoff(a, b)

        return rule.cutoff

    def is_default(self, a: str, b: str) -> bool:
        """Whether this pair still follows the covalent criterion."""

        rule = self.rule_for(a, b)

        return rule is None or (
            rule.enabled and rule.cutoff == FROM_RADII
        )

    def accepts(self, a: str, b: str, distance: float) -> bool:
        """Whether a contact at this distance is drawn as a bond."""

        return self.is_enabled(a, b) and distance <= self.cutoff(a, b)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def _rule(self, a: str, b: str) -> BondRule:
        """The rule for this pair, created at its defaults if need be."""

        key = pair_key(a, b)

        rule = self.rules.get(key)

        if rule is None:
            rule = BondRule()
            self.rules[key] = rule

        return rule

    def set_enabled(self, a: str, b: str, on: bool) -> None:
        """Draw this pair, or stop drawing it."""

        self._rule(a, b).enabled = bool(on)

    def set_cutoff(self, a: str, b: str, value: float) -> None:
        """
        Draw this pair out to ``value`` Angstroms.

        Setting it back to the covalent default is done with
        :meth:`clear_cutoff` rather than by passing the default's current
        number, so that the pair goes on tracking its radii instead of
        being pinned to whatever they happened to give.
        """

        self._rule(a, b).cutoff = max(MIN_CUTOFF, min(MAX_CUTOFF, value))

    def clear_cutoff(self, a: str, b: str) -> None:
        """Put this pair back on the covalent criterion, still drawn."""

        rule = self.rules.get(pair_key(a, b))

        if rule is not None:
            rule.cutoff = FROM_RADII

    def reset(self) -> None:
        """Forget every override."""

        self.rules.clear()

    def copy(self) -> BondRules:
        """
        An independent copy.

        The dialog edits a copy and hands it back, so that closing it
        without applying leaves the scene as it was.
        """

        return BondRules(
            scale=self.scale,
            rules={
                key: BondRule(rule.enabled, rule.cutoff)
                for key, rule in self.rules.items()
            },
        )

    # ------------------------------------------------------------------
    # For the bond builder
    # ------------------------------------------------------------------

    def cutoff_table(self, elements) -> dict[tuple[str, str], float]:
        """
        Squared cutoffs by pair, with the disabled pairs left out.

        Built once per redraw and read inside a loop over every pair of
        visible atoms, which is why it is squared: comparing against the
        square of the separation saves a square root per candidate, and
        on a large boundary there are tens of thousands of candidates.

        A pair that is switched off is absent rather than zero, so the
        builder skips it on a dictionary miss without a second test.
        """

        table: dict[tuple[str, str], float] = {}

        for a, b in element_pairs(elements):

            if not self.is_enabled(a, b):
                continue

            cutoff = self.cutoff(a, b)

            table[(a, b)] = cutoff * cutoff

        return table
