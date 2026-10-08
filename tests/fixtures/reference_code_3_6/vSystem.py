"""
Stochastic parametric L-system grammars for vascular trees.

Each function returns the string its rule produces after n iterations, in the
notation of Galarreta-Valverde (2012): f(length, diameter) moves the turtle,
+(theta) and -(theta) turn it, /(beta) rolls its perpendicular vector, [ and ]
push and pop the state, and { } delimit a stem that is interpolated as one
smooth segment. Operands are written with repr(float) so that the interpreter
reads them back exactly with float().

Provenance: F, S and D follow the tree grammar of section 4.3.1 of the
dissertation; A, B, I and R follow its example grammar (b). Departures from
the source are documented on each rule: stems are always drawn, so an
iteration count is a count of drawn generations; the length margin is
relative (see libGenerator); and anomalies are drawn per sub-segment with
configurable probabilities instead of being written into a grammar by hand.

The recursive rules also accept `d_min`, a smallest drawn diameter in grammar
units (micrometres by the convention of the README). A branch terminates as
soon as its diameter falls below it, so no vessel thinner than a modality's
smallest resolvable calibre is written into the grammar. Left at None the
rules stop on the iteration count alone; with both set, whichever comes first
wins. Diameters fall by 2^(-1/k) per generation, so the iteration count that
reaches d_min from d0 is about log(d0/d_min) / log(2^(1/k)). It bounds the
diameter a branch is drawn at, not the diameter after a local anomaly: a
stenosis still narrows a drawn sub-segment by stenosis_factor.

F can also fill the tree out with capillaries, a further departure from the
source. With `capillary_generations` m > 0, a daughter that F would end
because it fell below d_min is drawn instead as capillary_tree(m, d_min, E):
m generations of symmetric bifurcations at d_min. A daughter the iteration
count ends is not replaced, and the m generations do not use up the count.
With `capillary_runs` E, every capillary stem, and every stem of F whose two
daughters both end, is E consecutive S blocks in one pair of braces. At the
defaults (0 and 1) F is the rule above, draw for draw.
"""
import math
import numbers
import random

import libGenerator as lg
from libGenerator import calBifurcation, getLength


def _num(value):
    return repr(float(value))


def _segment(length, diameter=None):
    if diameter is None:
        return "f(" + _num(length) + ")"
    return "f(" + _num(length) + "," + _num(diameter) + ")"


def _symmetric_zamir_angle():
    # calBifurcation's expression for th1 at alpha = d2/d1 = 1, written out so
    # that the angle equals what calBifurcation returns for equal daughters bit
    # for bit without drawing anything. The expression is cubic in alpha
    # whatever k is, so this is acos(2^(-1/3)), about 37.467 degrees.
    alpha = 1.0
    xtmp = (1 + alpha ** 3) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha ** 3) ** (2.0 / 3))
    return math.acos(xtmp / xtmpb) * 180 / math.pi


# the turn into each daughter of a capillary tree, in degrees
_SYMMETRIC_ZAMIR_ANGLE = _symmetric_zamir_angle()


def F(n, d0, d_min=None, capillary_generations=0, capillary_runs=1):
    """
    Bifurcating tree: a stem, then two daughters turned by the Zamir angles
    on opposite sides of the parent, each followed by a roll of the
    perpendicular vector so that successive bifurcation planes differ.

    Source: F(d0) -> {S(d0)} [+(th1) /(70) F(d1)] [-(th2) /(70) F(d2)].
    The roll angle is the roll_angle property (70 degrees by default).

    The capillary fill is not in the source. With capillary_generations m > 0,
    a daughter thinner than d_min becomes capillary_tree(m, d_min, E) instead
    of "F", unless the iteration count has run out, which is checked first;
    a root thinner than d_min still gives "F". A stem is E S blocks long when
    both its daughters end as "F", and one block otherwise. A stem of F draws
    what it draws at the defaults, in the same order, plus 6 random.random and
    5 np.random.uniform calls for each extra block; a capillary tree draws
    what capillary_tree states.

    Args:
        n (int): remaining generations.
        d0 (float): diameter of this branch, in grammar units.
        d_min (float or None): smallest drawn diameter; a branch thinner than
            this terminates without drawing its stem, or, as a daughter with
            capillary generations, is drawn as a capillary tree.
        capillary_generations (int): m >= 0, the generations of each capillary
            tree. They are passed on whole and do not use up n. A positive m
            needs d_min.
        capillary_runs (int): E >= 1, the S blocks of every capillary stem and
            of every stem whose two daughters both end.

    Raises:
        ValueError: before anything is drawn, unless capillary_generations is
            an integer >= 0 and capillary_runs an integer >= 1 (numpy integers
            are accepted, bools and floats are not), and, when
            capillary_generations is positive, d_min is a finite positive
            number.
    """
    if not (type(capillary_generations) is int and capillary_generations == 0
            and type(capillary_runs) is int and capillary_runs == 1):
        # the plain 0 and 1 that the default rule passes on at every call
        # cannot be refused, and checking them at every call would make F
        # about 9% slower
        _check_fill(d_min, capillary_generations, capillary_runs)
    if capillary_generations == 0 and capillary_runs == 1:
        if n <= 0 or (d_min is not None and d0 < d_min):
            return "F"
        p = calBifurcation(d0)
        roll = _num(lg.roll_angle)
        return ("{" + S(d0) + "}"
                + "[+(" + _num(p["th1"]) + ")/(" + roll + ")" + F(n - 1, p["d1"], d_min) + "]"
                + "[-(" + _num(p["th2"]) + ")/(" + roll + ")" + F(n - 1, p["d2"], d_min) + "]")
    return _filled(n, d0, d_min, int(capillary_generations), int(capillary_runs))


def _check_fill(d_min, capillary_generations, capillary_runs):
    """Refuses F's malformed capillary fill settings, as F states, without drawing anything."""
    _check_integer("capillary_generations", capillary_generations, 0)
    _check_integer("capillary_runs", capillary_runs, 1)
    if capillary_generations > 0 and not _finite_positive(d_min):
        raise ValueError("capillary_generations > 0 needs d_min, a finite positive diameter, "
                         f"got {d_min!r}")


def capillary_tree(m, d, runs):
    """
    A capillary tree: m generations of symmetric bifurcations, all at one
    diameter. Each stem is `runs` S blocks in one pair of braces, and its two
    daughters are turned on opposite sides by Zamir's angle for equal
    daughters and rolled by roll_angle. Not part of the source grammar.

    Rule: capillary_tree(m, d) -> {S(d) ... S(d)} [+(th) /(roll) capillary_tree(m-1, d)]
                                                  [-(th) /(roll) capillary_tree(m-1, d)],
    and capillary_tree(m, d) -> F for m <= 0, where th = acos(2^(-1/3)),
    about 37.467 degrees, is the angle calBifurcation gives when d2 = d1.

    Every S block is evaluated afresh, with its own mirror form, lengths and
    anomalies, so a tree draws (2^m - 1) * runs * 6 random.random and
    (2^m - 1) * runs * 5 np.random.uniform calls and nothing else. Every turn
    and roll is written with its operand, so interpreting the tree draws
    nothing either.

    Args:
        m (int): generations; m <= 0 gives "F" whatever d and runs are.
        d (float): diameter of every stem, finite and positive.
        runs (int): S blocks per stem, at least 1.

    Raises:
        ValueError: before anything is drawn, for a non-integer or bool m,
            and, when m > 0, for a non-integer or bool runs, runs < 1, or a d
            that is not a finite positive number.
    """
    _check_integer("m", m)
    if m <= 0:
        return "F"
    _check_integer("runs", runs, 1)
    if not _finite_positive(d):
        raise ValueError(f"a capillary tree needs a finite positive diameter, got {d!r}")
    return _capillary_tree(int(m), d, int(runs))


def _check_integer(name, value, low=None):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or (low is not None and value < low)):
        bound = "" if low is None else f" >= {low}"
        raise ValueError(f"{name} must be an integer{bound}, got {value!r}")


def _finite_positive(value):
    return (not isinstance(value, bool) and isinstance(value, numbers.Real)
            and math.isfinite(value) and value > 0)


def _run(d0, blocks):
    # independent S evaluations: each block draws its own mirror form, lengths and anomalies
    return "".join(S(d0) for _ in range(blocks))


def _ends(n, d, d_min, m):
    """Whether the daughter F(n, d) of a filled tree is the terminal "F"."""
    return n <= 0 or (d_min is not None and d < d_min and m <= 0)


def _daughter(n, d, d_min, m, runs):
    # the iteration count is checked before d_min, so a daughter it ends stays "F"
    if n > 0 and d_min is not None and d < d_min and m > 0:
        return _capillary_tree(m, d_min, runs)
    return _filled(n, d, d_min, m, runs)


def _filled(n, d0, d_min, m, runs):
    """F with checked fill settings; draws in the order F draws."""
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "F"
    p = calBifurcation(d0)
    roll = _num(lg.roll_angle)
    blocks = runs if _ends(n - 1, p["d1"], d_min, m) and _ends(n - 1, p["d2"], d_min, m) else 1
    return ("{" + _run(d0, blocks) + "}"
            + "[+(" + _num(p["th1"]) + ")/(" + roll + ")" + _daughter(n - 1, p["d1"], d_min, m, runs) + "]"
            + "[-(" + _num(p["th2"]) + ")/(" + roll + ")" + _daughter(n - 1, p["d2"], d_min, m, runs) + "]")


def _capillary_tree(m, d, runs):
    if m <= 0:
        return "F"
    th = _num(_SYMMETRIC_ZAMIR_ANGLE)
    roll = _num(lg.roll_angle)
    return ("{" + _run(d, runs) + "}"
            + "[+(" + th + ")/(" + roll + ")" + _capillary_tree(m - 1, d, runs) + "]"
            + "[-(" + th + ")/(" + roll + ")" + _capillary_tree(m - 1, d, runs) + "]")


def S(d0):
    """
    Stem of five sub-segments with alternating turns of stem_angle degrees;
    the two mirror-image forms S1 and S2 are chosen with equal probability.

    Source: S(d0):0.5 -> D +(25) D -(25) D -(25) D +(25) D and its mirror.
    """
    return S1(d0) if random.random() < 0.5 else S2(d0)


def S1(d0):
    a = _num(lg.stem_angle)
    return (D(d0) + "+(" + a + ")" + D(d0) + "-(" + a + ")" + D(d0)
            + "-(" + a + ")" + D(d0) + "+(" + a + ")" + D(d0))


def S2(d0):
    a = _num(lg.stem_angle)
    return (D(d0) + "-(" + a + ")" + D(d0) + "+(" + a + ")" + D(d0)
            + "+(" + a + ")" + D(d0) + "-(" + a + ")" + D(d0))


def D(d0, divisor=5.0):
    """
    One stem sub-segment of length co/divisor at diameter d0. With probability
    aneurysm_prob or stenosis_prob the sub-segment carries a local change of
    diameter over its middle three fifths, as in the anomaly grammars (i) and
    (j) of the source.

    Source: D(d0) -> f('co/5', d0), and
            D(d0):0.3 -> f('co/25', d0) f('3*co/25', 'd0*1.5') f('co/25', d0).
    """
    length = getLength(d0) / divisor
    r = random.random()
    if r < lg.aneurysm_prob:
        return _anomaly(length, d0, lg.aneurysm_factor)
    if r < lg.aneurysm_prob + lg.stenosis_prob:
        return _anomaly(length, d0, lg.stenosis_factor)
    return _segment(length, d0)


def _anomaly(length, d0, factor):
    end = length / 5.0
    return (_segment(end, d0)
            + _segment(3.0 * length / 5.0, d0 * factor)
            + _segment(end, d0))


def C(d0, divisor=7.0, bend=18.0):
    """
    A gently curving sub-segment used by the side-branch grammar.

    Source: D(d0) -> f('co/7', d0) -(18) in example (b).
    """
    return _segment(getLength(d0) / divisor, d0) + "-(" + _num(bend) + ")"


def A(n, d0, d_min=None):
    """
    Bifurcating sub-tree without rolls, used by the side-branch grammar.

    Source: A(d0) -> S(d0) [+(th1) A(d1)] [-(th2) A(d2)].
    """
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "A"
    p = calBifurcation(d0)
    return ("{" + S(d0) + "}"
            + "[+(" + _num(p["th1"]) + ")" + A(n - 1, p["d1"], d_min) + "]"
            + "[-(" + _num(p["th2"]) + ")" + A(n - 1, p["d2"], d_min) + "]")


def B(n, d0, d_min=None):
    """
    Three curving sub-segments, a roll of 90 degrees, then a sub-tree.

    Source: B(d0) -> D(d0) D(d0) D(d0) /(90) A(d0).
    """
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "B"
    return C(d0) + C(d0) + C(d0) + "/(90.0)" + A(n - 1, d0, d_min)


def R(n, d0, d_min=None):
    """
    A segment with a side branch and a continuing trunk.

    Source: R(d0) -> f('co/3') D D D [B(d1)] f('co/2', d2) B(d2).
    """
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "R"
    p = calBifurcation(d0)
    # the trunk carries on at d2, so it is subject to d_min like any other branch
    trunk = ""
    if d_min is None or p["d2"] >= d_min:
        trunk = _segment(p["co"] / 2.0, p["d2"]) + B(n - 1, p["d2"], d_min)
    return (_segment(p["co"] / 3.0) + C(d0) + C(d0) + C(d0)
            + "[" + B(n - 1, p["d1"], d_min) + "]" + trunk)


def I(n, d0, d_min=None):
    """
    Initial segment followed by one side branch.

    Source: I(d0) -> f('co/3', d0) +(25) [R(d0)].
    """
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "I"
    return (_segment(getLength(d0) / 3.0, d0) + "+(" + _num(lg.stem_angle) + ")"
            + "[" + R(n - 1, d0, d_min) + "]")


def example_grammar(n, d0):
    """
    The example rule of the 2013 paper (Table 4.2): a bare f that takes the
    default length and diameter, then two daughters turned by the Zamir angles.

    Source: F(d0) -> f [+(th1) F(d1)] [-(th2) F(d2)].
    """
    if n <= 0:
        return "F"
    p = calBifurcation(d0)
    return ("f[+(" + _num(p["th1"]) + ")" + example_grammar(n - 1, p["d1"]) + "]"
            + "[-(" + _num(p["th2"]) + ")" + example_grammar(n - 1, p["d2"]) + "]")
