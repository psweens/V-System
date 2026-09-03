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
"""
import random

import libGenerator as lg
from libGenerator import calBifurcation, getLength


def _num(value):
    return repr(float(value))


def _segment(length, diameter=None):
    if diameter is None:
        return "f(" + _num(length) + ")"
    return "f(" + _num(length) + "," + _num(diameter) + ")"


def F(n, d0, d_min=None):
    """
    Bifurcating tree: a stem, then two daughters turned by the Zamir angles
    on opposite sides of the parent, each followed by a roll of the
    perpendicular vector so that successive bifurcation planes differ.

    Source: F(d0) -> {S(d0)} [+(th1) /(70) F(d1)] [-(th2) /(70) F(d2)].
    The roll angle is the roll_angle property (70 degrees by default).

    Args:
        n (int): remaining generations.
        d0 (float): diameter of this branch, in grammar units.
        d_min (float or None): smallest drawn diameter; a branch thinner than
            this terminates without drawing its stem.
    """
    if n <= 0 or (d_min is not None and d0 < d_min):
        return "F"
    p = calBifurcation(d0)
    roll = _num(lg.roll_angle)
    return ("{" + S(d0) + "}"
            + "[+(" + _num(p["th1"]) + ")/(" + roll + ")" + F(n - 1, p["d1"], d_min) + "]"
            + "[-(" + _num(p["th2"]) + ")/(" + roll + ")" + F(n - 1, p["d2"], d_min) + "]")


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
