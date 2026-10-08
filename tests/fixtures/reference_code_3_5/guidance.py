"""
Directional guidance of the walk: a deterministic drift of the heading
towards an axis or a plane, chosen per calibre class.

The persistent random walk (tortuosity.py) diffuses the heading of a stem:
every step of length h rotates it by an angle drawn from N(0, sqrt(h / (P d)))
about a random axis perpendicular to it, where d is the vessel diameter and
P the persistence in diameters. Guidance adds a drift before each step's
draw: the heading h is turned towards a target t by an angle

    omega = min(theta, r sin(theta) cos(theta))   for a nematic axis or a plane,
    omega = min(theta, r sin(theta))              for a polar axis,

where theta is the angle from h to t and r = min(1, step / (G d)), G the
guidance length in local diameters. The drift draws nothing, so with
guidance off the walk draws exactly what it drew before, and with guidance
on it still draws two numbers per attempt.

The target depends on the field of the rule that governs the stem:

    nematic axis a:  t = a when h . a >= 0, else -a (the nearer end);
    polar axis a:    t = s a, with s fixed per tree: +1 ("fixed"), the sign
                     of the tree's root heading along a ("root"), or the sign
                     along a of the partner tree's root offset minus this
                     tree's, v_partner - v_own with v the root_offsets
                     entries ("partner");
    plane, normal n: t = the projection of h onto the plane, undefined when
                     h is parallel to n.

A plane rule may also "bank": after the turn, the frame's perpendicular is
turned towards the plane normal by the same rule, so that the branching
turns the grammar writes about the perpendicular stay in the plane despite
the roll written before every daughter.

The law. The walk diffuses the heading with D = 1 / (4 P d) per unit length,
and the drift descends the potential Phi = -cos^2(theta) / 2 (nematic),
-cos(theta) (polar) or sin^2(beta) / 2 (plane, beta the angle out of the
plane), each divided by G d per unit length. On a stem much longer than G d
the heading therefore settles into a stationary law:

    nematic:  Watson,  exp(K cos^2 theta)  with K = 2 P / G;
    polar:    Fisher,  exp(kappa cos theta) with kappa = 4 P / G;
    plane:    girdle,  exp(-K sin^2 beta)   with K = 2 P / G,

with small-angle rms angles of sqrt(G / (2 P)) to an axis and sqrt(G / (4 P))
out of a plane. G follows from a target K. The steps are finite, so the
measured concentration carries a discretisation bias of order step / (G d).
The nematic law is Watson, not Fisher-axial: Fisher-axial is exp(K |cos
theta|), with <|cos theta|> = coth K - 1 / K, which is the polar law's
<cos theta> at kappa = K up to the polar law's backward mass.

Rules. A guidance specification is a list of JSON-ready rule dicts, each
governing the stems whose diameter at the first move lies below `below`
times d_min (None: any diameter) and above the bound of the rule before it,
so the rules partition the calibres from the finest upwards:

    below     class bound in d_min, exclusive; None = no bound     required
    field     "axis", "plane", or None (an unguided class)          required
    axis      a 3-vector (axis rules)                               required with an axis
    normal    a 3-vector (plane rules)                              required with a plane
    sense     "nematic" or "polar" (axis rules)                     nematic
    polarity  "root", "fixed" or "partner" (polar axis rules)       root
    length    G, in local diameters                                 required with a field
    onset     in local diameters                                    2.0
    bank      a bool (plane rules)                                  false

A stem is not steered while the arc walked along it is below onset times its
diameter, so that daughters clear their sisters before being pulled into
line. A stem below no bound is unguided. The rule is chosen once per walked
polyline, from the diameter of its first move, and held, so that a local
anomaly does not switch class part way along.

Events. Every call of `steer` counts exactly one of guided, onset, unguided
(no rule, or a rule without a field) and undefined (a plane target with the
heading along the normal) steps, whether or not the step is then accepted.
Sense flips count the changes of the nearer end of a nematic axis between
consecutive guided steps of one polyline; bank steps count bank rotations
greater than zero.

Everything here is deterministic and draws nothing. Rotations use the
Rodrigues form of tortuosity._rotate. Python 3.9 compatible; numpy only.
"""
import math
import numbers

import numpy as np

from tortuosity import _rotate

# The fields a rule may steer towards, the senses of an axis and the ways a
# polar sense is fixed per tree. Append-only: a recorded frame names them.
FIELDS = ("axis", "plane")
SENSES = ("nematic", "polar")
POLARITIES = ("root", "fixed", "partner")

# The counters guidance reports, appended to main.EVENT_KEYS.
EVENT_KEYS = ("guided_steps", "guidance_onset_steps", "unguided_steps", "guidance_undefined_steps",
              "guidance_sense_flips", "guidance_bank_steps")

# Arc walked along a stem, in local diameters, before it is steered.
DEFAULT_ONSET = 2.0

FRAME_VERSION = 1

# The keys a normalised rule holds, in order, by field.
_AXIS_KEYS = ("below", "field", "axis", "sense", "polarity", "length", "onset")
_PLANE_KEYS = ("below", "field", "normal", "length", "onset", "bank")
_NONE_KEYS = ("below", "field")
_KEYS_BY_FIELD = {"axis": _AXIS_KEYS, "plane": _PLANE_KEYS, None: _NONE_KEYS}

_TINY = 1e-12


def _is_number(value):
    # any real number but a bool, numpy scalars included; an integer too large
    # for a float is not finite
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _unit_vector(value, name, index):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_is_number(v) for v in value):
        raise ValueError(f"guidance rule {index}: {name} must be three finite numbers")
    vector = np.asarray(value, dtype=float)
    # the norm is taken of the vector scaled by its largest component, which
    # cannot overflow, and scaled back as a Python float, which may reach inf
    largest = float(np.max(np.abs(vector)))
    scaled = vector / largest if largest > 0.0 else vector
    norm = largest * float(np.linalg.norm(scaled))
    if norm <= _TINY:
        raise ValueError(f"guidance rule {index}: {name} must not be a zero vector")
    # a vector that is already unit is stored as it is, so that normalising a
    # recorded rule again returns the recorded floats
    if abs(norm - 1.0) > _TINY:
        vector = scaled / np.linalg.norm(scaled)
    return [float(v) for v in vector]


def parse_rules(spec, d_min):
    """
    Validates a guidance specification and returns its normalised rules.

    Args:
        spec (list): rule dicts as described in the module docstring.
        d_min (float or None): the unit of `below`; a finite bound needs it.

    Returns:
        list: new dicts, one per rule, holding exactly the keys of their field
        in a fixed order: `below` (float or None), `field`, then for an axis
        `axis` (a unit list of three floats), `sense`, `polarity` (polar axes
        only), `length` and `onset`; for a plane `normal` (unit), `length`,
        `onset` and `bank`. Parsing a normalised list returns an equal list.

    Raises:
        ValueError: for anything the module docstring does not allow.
    """
    if not isinstance(spec, list) or len(spec) == 0:
        raise ValueError("guidance must be a non-empty list of rules")
    rules = []
    previous = None
    for index, rule in enumerate(spec):
        if not isinstance(rule, dict):
            raise ValueError(f"guidance rule {index} is not a dict")
        if "field" not in rule or "below" not in rule:
            raise ValueError(f"guidance rule {index} needs 'below' and 'field'")
        field = rule["field"]
        if field is not None and field not in FIELDS:
            raise ValueError(f"guidance rule {index}: field must be one of {FIELDS} or None, got {field!r}")
        allowed = _KEYS_BY_FIELD[field]
        for key in rule:
            if key not in allowed:
                where = f"a {field} rule" if field is not None else "a rule without a field"
                raise ValueError(f"guidance rule {index}: unknown key {key!r} for {where}")
        below = rule["below"]
        if below is not None:
            if not _is_number(below) or below <= 0.0:
                raise ValueError(f"guidance rule {index}: below must be a positive number or None")
            if d_min is None:
                raise ValueError(f"guidance rule {index}: a finite class bound needs d_min")
            below = float(below)
            if previous is not None and below <= previous:
                raise ValueError("guidance rules must have strictly increasing bounds")
        elif index != len(spec) - 1:
            raise ValueError("only the last guidance rule may have below None")
        previous = below
        normalised = {"below": below, "field": field}
        if field is None:
            rules.append(normalised)
            continue
        if field == "axis":
            if "axis" not in rule:
                raise ValueError(f"guidance rule {index}: an axis rule needs an axis")
            normalised["axis"] = _unit_vector(rule["axis"], "axis", index)
            sense = rule.get("sense", "nematic")
            if sense not in SENSES:
                raise ValueError(f"guidance rule {index}: sense must be one of {SENSES}, got {sense!r}")
            normalised["sense"] = sense
            if sense == "polar":
                polarity = rule.get("polarity", "root")
                if polarity not in POLARITIES:
                    raise ValueError(f"guidance rule {index}: polarity must be one of {POLARITIES}, got {polarity!r}")
                normalised["polarity"] = polarity
            elif "polarity" in rule:
                raise ValueError(f"guidance rule {index}: polarity belongs to a polar axis")
        else:
            if "normal" not in rule:
                raise ValueError(f"guidance rule {index}: a plane rule needs a normal")
            normalised["normal"] = _unit_vector(rule["normal"], "normal", index)
        if "length" not in rule or not _is_number(rule["length"]) or rule["length"] <= 0.0:
            raise ValueError(f"guidance rule {index}: a field needs a positive length")
        normalised["length"] = float(rule["length"])
        onset = rule.get("onset", DEFAULT_ONSET)
        if not _is_number(onset) or onset < 0.0:
            raise ValueError(f"guidance rule {index}: onset must be a non-negative number")
        normalised["onset"] = float(onset)
        if field == "plane":
            bank = rule.get("bank", False)
            if not isinstance(bank, bool):
                raise ValueError(f"guidance rule {index}: bank must be a bool")
            normalised["bank"] = bank
        rules.append(normalised)
    return rules


def _orthonormalise(heading, perp):
    """The frame after a rotation, re-orthonormalised as tortuosity.perpendicular_rotation does."""
    heading = heading / np.linalg.norm(heading)
    perp = perp - (perp @ heading) * heading
    perp = perp / np.linalg.norm(perp)
    return heading, perp


def guide_frame(heading, perp, target, rate, axial):
    """
    Turns the turtle frame towards a unit target by the drift of one step.

    Args:
        heading, perp (ndarray): the unit frame.
        target (ndarray): the unit target direction t.
        rate (float): r = min(1, step / (G d)).
        axial (bool): the nematic and plane law omega = min(theta, r sin theta
            cos theta), which drifts to the nearer of t and -t; False gives
            the polar law omega = min(theta, r sin theta).

    Returns:
        tuple: (heading, perp, omega), the frame rotated by omega about
        h x t and re-orthonormalised. The inputs come back unchanged, with
        omega 0, when |h x t| <= 1e-12: aligned, or a polar target directly
        behind, which exerts no torque.
    """
    cross = np.cross(heading, target)
    sine = float(np.linalg.norm(cross))
    if sine <= _TINY:
        return heading, perp, 0.0
    cosine = float(heading @ target)
    theta = math.atan2(sine, cosine)
    omega = min(theta, rate * sine * cosine) if axial else min(theta, rate * sine)
    if omega == 0.0:
        return heading, perp, 0.0
    axis = cross / sine
    heading, perp = _orthonormalise(_rotate(heading, axis, omega), _rotate(perp, axis, omega))
    return heading, perp, omega


def bank_frame(heading, perp, normal, rate):
    """
    Turns the perpendicular towards the plane normal, about the heading, by
    the nematic drift of one step: psi = angle(perp, t_p) with t_p the
    normal's component perpendicular to the heading, signed so that
    perp . t_p >= 0, and the rotation min(psi, r sin psi cos psi).

    Returns:
        tuple: (perp, omega); the inputs unchanged with omega 0 when the
        heading is along the normal (|t_p| < 1e-12) or perp is already
        along t_p (|perp x t_p| <= 1e-12).
    """
    component = normal - (normal @ heading) * heading
    size = float(np.linalg.norm(component))
    if size < _TINY:
        return perp, 0.0
    target = component / size
    if perp @ target < 0.0:
        target = -target
    cross = np.cross(perp, target)
    sine = float(np.linalg.norm(cross))
    if sine <= _TINY:
        return perp, 0.0
    cosine = float(perp @ target)
    psi = math.atan2(sine, cosine)
    omega = min(psi, rate * sine * cosine)
    if omega == 0.0:
        return perp, 0.0
    perp = _rotate(perp, cross / sine, omega)
    perp = perp - (perp @ heading) * heading
    return perp / np.linalg.norm(perp), omega


class _Compiled:
    """A rule's vectors and numbers as arrays and floats, for the walk's inner loop."""

    __slots__ = ("field", "vector", "polar", "polarity", "length", "onset", "bank", "sign")

    def __init__(self, rule):
        self.field = rule["field"]
        self.vector = None
        self.polar = False
        self.polarity = None
        self.length = self.onset = None
        self.bank = False
        self.sign = None            # +1 or -1 once a polar rule is bound to a tree
        if self.field == "axis":
            self.vector = np.asarray(rule["axis"], dtype=float)
            self.polar = rule["sense"] == "polar"
            self.polarity = rule.get("polarity")
            if self.polarity == "fixed":
                self.sign = 1.0
        elif self.field == "plane":
            self.vector = np.asarray(rule["normal"], dtype=float)
            self.bank = rule["bank"]
        if self.field is not None:
            self.length = rule["length"]
            self.onset = rule["onset"]


class Guidance:
    """
    The guidance rules of one network, and their application to a walk.

    Args:
        rules (list): a guidance specification (see parse_rules) or the
            normalised rules of a frame record.
        d_min (float or None): the unit of the class bounds.

    The instance a network is grown with is bound per tree (`bind`), which
    fixes the sense of every polar rule, and the per-tree copies are what the
    walk steers with. `rules` holds the normalised rules.
    """

    def __init__(self, rules, d_min):
        self.rules = parse_rules(rules, d_min)
        self.d_min = None if d_min is None else float(d_min)
        self._compiled = [_Compiled(rule) for rule in self.rules]
        self._slots = {id(rule): k for k, rule in enumerate(self.rules)}
        self._last_sign = 0.0      # the nearer end of a nematic axis at the last guided step
        self.bound = False

    def rule_for(self, diameter):
        """The first rule whose bound the diameter lies below (None bounds any), or None."""
        for rule in self.rules:
            below = rule["below"]
            if below is None or diameter < below * self.d_min:
                return rule
        return None

    def bind(self, root_heading, partner_offset=None):
        """
        A copy of the guidance for one tree, with every polar sense fixed:
        +1 for "fixed", the sign of root_heading . a (+1 when 0) for "root",
        and the sign of partner_offset . a for "partner".

        Raises:
            ValueError: for a "partner" rule without a partner offset, or
            with an offset perpendicular to the axis.
        """
        root_heading = np.asarray(root_heading, dtype=float)
        bound = Guidance.__new__(Guidance)
        bound.rules = self.rules
        bound.d_min = self.d_min
        bound._slots = self._slots
        bound._last_sign = 0.0
        bound.bound = True
        bound._compiled = []
        for rule in self._compiled:
            copy = _Compiled.__new__(_Compiled)
            for name in _Compiled.__slots__:
                setattr(copy, name, getattr(rule, name))
            if copy.polar and copy.polarity == "root":
                copy.sign = -1.0 if float(root_heading @ copy.vector) < 0.0 else 1.0
            elif copy.polar and copy.polarity == "partner":
                if partner_offset is None:
                    raise ValueError("a 'partner' polarity needs a second tree and root_offsets")
                along = float(np.asarray(partner_offset, dtype=float) @ copy.vector)
                if along == 0.0:
                    raise ValueError("a 'partner' polarity needs root offsets that differ along the axis")
                copy.sign = 1.0 if along > 0.0 else -1.0
            bound._compiled.append(copy)
        return bound

    def _compiled_for(self, rule):
        try:
            return self._compiled[self._slots[id(rule)]]
        except KeyError:
            return self._compiled[self.rules.index(rule)]

    def steer(self, heading, perp, pos, rule, diameter, step, arc, events):
        """
        Steers the frame for one walk step under `rule` (as rule_for returns
        it, or None), counting the step in `events`.

        Args:
            heading, perp (ndarray): the unit frame before the step.
            pos (ndarray): the position before the step (unused by the fields
                of this version, which do not depend on position).
            rule (dict or None): the stem's rule.
            diameter (float): the move's diameter d.
            step (float): the step length h.
            arc (float): the arc already walked along this polyline; 0 at its
                first step, which resets the sense-flip record.
            events (dict): the counters, incremented in place.

        Returns:
            tuple: (heading, perp).
        """
        if arc <= 0.0:
            self._last_sign = 0.0
        if rule is None or rule["field"] is None:
            events["unguided_steps"] += 1
            return heading, perp
        compiled = self._compiled_for(rule)
        if arc < compiled.onset * diameter:
            events["guidance_onset_steps"] += 1
            return heading, perp
        rate = min(1.0, step / (compiled.length * diameter))
        if compiled.field == "axis":
            if compiled.polar:
                if compiled.sign is None:
                    raise ValueError("a polar rule must be bound to a tree before it steers")
                target = compiled.sign * compiled.vector
                heading, perp, _ = guide_frame(heading, perp, target, rate, False)
            else:
                sign = -1.0 if float(heading @ compiled.vector) < 0.0 else 1.0
                if self._last_sign and sign != self._last_sign:
                    events["guidance_sense_flips"] += 1
                self._last_sign = sign
                heading, perp, _ = guide_frame(heading, perp, sign * compiled.vector, rate, True)
            events["guided_steps"] += 1
            return heading, perp
        normal = compiled.vector
        projection = heading - (heading @ normal) * normal
        size = float(np.linalg.norm(projection))
        if size < _TINY:
            events["guidance_undefined_steps"] += 1
            return heading, perp
        heading, perp, _ = guide_frame(heading, perp, projection / size, rate, True)
        events["guided_steps"] += 1
        if compiled.bank:
            perp, omega = bank_frame(heading, perp, normal, rate)
            if omega > 0.0:
                events["guidance_bank_steps"] += 1
        return heading, perp

    def frame_record(self, direction, perpendicular, origin, bound):
        """
        The frame record of a network grown with these rules: kind, axis,
        normal, sense and the per-tree senses of the field rule with the
        smallest bound, every rule, and the growth frame.

        Args:
            direction, perpendicular: the turtle frame of the first tree.
            origin: the growth-box centre, else the first root.
            bound (sequence): the per-tree copies from `bind`, in tree order.
        """
        record = unguided_frame(direction, perpendicular, origin)
        record["rules"] = [dict(rule) for rule in self.rules]
        for k, rule in enumerate(self.rules):
            if rule["field"] is None:
                continue
            record["kind"] = rule["field"]
            if rule["field"] == "axis":
                record["axis"] = list(rule["axis"])
                record["sense"] = rule["sense"]
                if rule["sense"] == "polar":
                    record["tree_senses"] = [int(tree._compiled[k].sign) for tree in bound]
            else:
                record["normal"] = list(rule["normal"])
            break
        return record


def unguided_frame(direction, perpendicular, origin):
    """The frame record of a network grown without guidance: kind "none" and no rules."""
    return {"frame_version": FRAME_VERSION, "kind": "none", "axis": None, "sense": None, "tree_senses": None,
            "normal": None, "origin": [float(v) for v in origin],
            "grow_direction": [float(v) for v in direction],
            "grow_perpendicular": [float(v) for v in perpendicular], "rules": []}
