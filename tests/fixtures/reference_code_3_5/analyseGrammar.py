"""
Turtle interpretation of V-System programs.

Implements the interpreter of Galarreta-Valverde (2012, sections 3.1.1 and
3.3.1). The turtle state is a position, a unit direction vector, a unit
perpendicular vector and a diameter:

    f(l, d)  move by l along the direction and record diameter d; a missing
             d keeps the current diameter, a missing l is the default segment
             length for the current diameter
    +(t)     rotate the direction about the perpendicular by t degrees
    -(t)     the same, clockwise; a missing t is the Zamir angle theta1 for
             the current diameter
    /(b)     rotate the perpendicular about the direction by b degrees
    *(b)     the same, clockwise; a missing b is the roll_angle property
    [ ]      push and pop the whole state
    { }      delimit a segment whose points are interpolated as one curve

A '[' met while a segment is open closes it, and the matching ']' opens a
fresh one for the continuation. The stem is thereby pinned at the branch
point: the part before it is one curve ending exactly there, the daughter
starts exactly there, and the rest of the stem is a second curve starting
exactly there. Had the daughter's points been folded into the parent's
segment instead, the interpolating spline would have run past the branch
point without touching it and the continuation would have restarted from a
point the curve never visits, leaving the centreline disconnected.

Upper-case letters are non-terminals left over from the recursion and draw
nothing. Whitespace between tokens is ignored.

Two optional shaping modes change how a braced stem is realised without
changing what the grammar decides, and the walk may in turn be guided. With a `walk` (see tortuosity.py) the moves
inside a stem are replaced by a persistent random walk of the same arc length,
so the stem curves continuously instead of zig-zagging, and the turns written
inside the stem are ignored; the turtle's position and frame after the stem are
the walk's. With an `avoid` (see collisions.py) every point is checked against
the network placed so far: a walk step that collides is redrawn up to a budget,
each redraw turning more sharply than the last so the walk steers clear,
a stem drawn as a spline is shortened to its longest collision-free prefix, and
a branch that cannot be placed is terminated and counted. In either mode the
interpreter samples the stem itself and yields plain polyline rows, so that
utils.interpolate_segments has nothing left to smooth; the samples of an
unshortened stem are identical to those the default path produces. Neither
mode is used unless asked for, and the default path draws no random numbers.

A walk may carry a `guidance` (see guidance.py): before every step's draw
the heading is turned deterministically towards an axis or a plane under
the rule chosen for the stem's calibre. The rule is chosen once per walked
polyline, from the diameter of its first move, and held, so that a local
anomaly does not switch class part way along; bare moves and spline stems
are never steered. Each redraw of a step draws its noise about the steered
frame, so the walk still draws two numbers per attempt, and without a
guidance nothing else runs.
"""
import math
import re

import numpy as np

import libGenerator as lg
from libGenerator import calBifurcation, getLength
from guidance import EVENT_KEYS as GUIDANCE_EVENTS
from utils import bspline, rotate_about, unit
from tortuosity import perpendicular_rotation, step_std

# A token is a single-character command, optionally followed by a parenthesised,
# comma-separated operand list: 'f(46.1,20.0)', '+(-37.5)', '[', '{'.
_TOKEN = re.compile(r"(?P<cmd>[A-Za-z\[\]{}+\-/*])(?:\((?P<args>[^()]*)\))?")

MOVE_COMMANDS = frozenset("f")

# Field order of the rows yielded by branching_turtle_to_coords.
ROW = ("x", "y", "z", "diameter", "segment")

_NAN_ROW = (math.nan,) * 5


def tokenise(turtle_program):
    """
    Splits a turtle program into (command, operands) pairs.

    Operands are converted with float(), so signs, decimals and exponent
    notation ('3.9e-05') are read as numbers rather than scanned as commands.
    Whitespace between tokens is ignored.

    Raises:
    ValueError: on unexpected characters, an empty or non-numeric operand,
                or a non-finite operand.
    """
    pos = 0
    for match in _TOKEN.finditer(turtle_program):
        gap = turtle_program[pos:match.start()]
        if gap and not gap.isspace():
            raise ValueError(
                f"unexpected text {gap!r} at position {pos} of the turtle program")
        pos = match.end()
        args = match.group("args")
        if args is None:
            params = ()
        else:
            try:
                params = tuple(float(a) for a in args.split(","))
            except ValueError as exc:
                raise ValueError(
                    f"could not parse operands {args!r} of command "
                    f"{match.group('cmd')!r} at position {match.start()}") from exc
            if not all(math.isfinite(v) for v in params):
                raise ValueError(
                    f"non-finite operand in {match.group(0)!r} "
                    f"at position {match.start()}")
        yield match.group("cmd"), params
    tail = turtle_program[pos:]
    if tail and not tail.isspace():
        raise ValueError(f"unexpected text {tail!r} at position {pos} of the turtle program")


def _angle(params, default):
    return params[0] if params else default()


def branching_turtle_to_coords(turtle_program, d0,
                               position=(0.0, 0.0, 0.0),
                               direction=(0.0, 1.0, 0.0),
                               perpendicular=(0.0, 0.0, 1.0),
                               bounds=None, walk=None, avoid=None, events=None, subdivisions=3):
    """
    Interprets a turtle program and yields its points.

    Args:
        turtle_program (str): the L-system string.
        d0 (float): initial diameter.
        position, direction, perpendicular: initial state; direction and
            perpendicular must be orthogonal. The defaults are the source's.
        bounds (sequence or None): ((x0, y0, z0), (x1, y1, z1)) in grammar
            units. A move that would leave the box terminates its branch: the
            move and everything descending from it are skipped, up to the ']'
            that closes the branch. The tree therefore grows within the box and
            no vessel is ever cut part way along, unlike a network rendered
            larger than its volume and cropped. None grows without limit.

    Yields:
        tuple: (x, y, z, diameter, segment). `segment` is the index of the
        enclosing {...} segment, or -1 outside braces. The initial position
        is yielded first. A row of NaN marks the end of a branch (a ']'),
        after which the restored point is yielded again -- with a fresh
        segment index if the branch was opened inside a segment, so that the
        continuation is interpolated as its own curve from the branch point.

        walk (WalkSettings or None): replace the moves inside every braced
            stem by a persistent random walk of the same arc length; see
            tortuosity.py. The rows of a walked stem carry segment -1, since
            they are already the final samples.
        avoid (collisions.CollisionAvoider or None): reject points that
            collide with the network placed so far, redrawing walk steps and
            shortening spline stems; the avoider's index is filled as the
            interpretation proceeds. Also yields final samples (segment -1).
        events (dict or None): counters incremented in place: a branch that
            a bound terminates ("bound_terminations"), walk steps redrawn or
            branches terminated at the bounds ("walk_bound_redraws",
            "walk_bound_terminations"), and collision redraws, terminations
            and shortened stems ("collision_redraws",
            "collision_terminations", "collision_truncated_stems"). None
            counts nothing.
        subdivisions (int): sampling depth used when the interpreter samples
            stems itself, 2**subdivisions points per sub-segment; unused on
            the default path, where utils.interpolate_segments samples them.

    Raises:
        ValueError: on a malformed program, unbalanced brackets, an empty box,
            or a starting position outside it.
    """
    if walk is None and avoid is None:
        yield from _interpret(turtle_program, d0, position, direction, perpendicular, bounds, events)
    else:
        yield from _interpret_shaped(turtle_program, d0, position, direction, perpendicular, bounds,
                                     walk, avoid, events, subdivisions)


class WalkSettings:
    """
    Settings of the persistent random walk that shapes a stem.

    Args:
        persistence (float): persistence length in multiples of the local
            vessel diameter; see tortuosity.py for the rotation rule.
        rng (numpy.random.Generator): the walk's own generator, seeded from
            the run seed by the caller so that the grammar's draws are
            untouched.
        bound_attempts (int): redraws a step that would leave the growth box
            is allowed before the branch is terminated.
        guidance (guidance.Guidance or None): a per-tree guidance that
            steers every walk step towards its rule's field; None leaves the
            walk as it is.
    """

    def __init__(self, persistence, rng, bound_attempts=20, guidance=None):
        if not persistence > 0.0:
            raise ValueError(f"persistence must be a positive number of diameters, got {persistence!r}")
        if bound_attempts < 0:
            raise ValueError("bound_attempts cannot be negative")
        self.persistence = float(persistence)
        self.rng = rng
        self.bound_attempts = int(bound_attempts)
        self.guidance = guidance


def _start_state(d0, position, direction, perpendicular, bounds):
    pos = np.asarray(position, dtype=float)
    heading = unit(direction)
    perp = unit(perpendicular)
    if abs(heading @ perp) > 1e-9:
        raise ValueError("direction and perpendicular must be orthogonal")
    diam = float(d0)
    if not diam > 0:
        raise ValueError(f"the initial diameter must be positive, got {d0!r}")
    low = high = None
    if bounds is not None:
        low, high = (np.asarray(b, dtype=float) for b in bounds)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError("bounds must be ((x0, y0, z0), (x1, y1, z1)) with x1 > x0 and so on")
        if np.any(pos < low) or np.any(pos > high):
            raise ValueError(f"the starting position {tuple(pos)} is outside the bounds")
    return pos, heading, perp, diam, low, high


def _interpret(turtle_program, d0, position, direction, perpendicular, bounds, events):
    """The default interpretation: control points for utils.interpolate_segments."""
    pos, heading, perp, diam, low, high = _start_state(d0, position, direction, perpendicular, bounds)

    stack = []
    segment = -1
    next_segment = 0
    depth = 0
    pruned_at = None      # the bracket depth of the branch a bound terminated, if any
    yield (pos[0], pos[1], pos[2], diam, segment)

    for command, params in tokenise(turtle_program):
        # a terminated branch draws nothing until the ']' that closes it, but its
        # brackets are still counted so that the matching one is recognised
        if pruned_at is not None and command not in ("[", "]"):
            continue

        if command in MOVE_COMMANDS:
            length = params[0] if params else getLength(diam)
            if len(params) > 1 and params[1] > 0.0:
                diam = params[1]
            step = pos + length * heading
            if low is not None and (np.any(step < low) or np.any(step > high)):
                pruned_at = depth       # this branch leaves the box: drop it and its subtree
                if events is not None:
                    events["bound_terminations"] = events.get("bound_terminations", 0) + 1
                continue
            pos = step
            yield (pos[0], pos[1], pos[2], diam, segment)

        elif command == "+":
            heading = rotate_about(heading, perp, _angle(params, lambda: calBifurcation(diam)["th1"]))
        elif command == "-":
            heading = rotate_about(heading, perp, -_angle(params, lambda: calBifurcation(diam)["th1"]))
        elif command == "/":
            perp = rotate_about(perp, heading, _angle(params, lambda: lg.roll_angle))
        elif command == "*":
            perp = rotate_about(perp, heading, -_angle(params, lambda: lg.roll_angle))

        elif command == "[":
            depth += 1
            if pruned_at is not None:
                continue
            stack.append((pos, heading, perp, diam, segment))
            segment = -1  # a branch leaving an open stem pins the stem here
        elif command == "]":
            depth -= 1
            if pruned_at is not None:
                if depth >= pruned_at:
                    continue            # still inside the terminated branch
                pruned_at = None        # the branch ends here, so drawing resumes
            if not stack:
                raise ValueError("']' without a matching '['")
            pos, heading, perp, diam, opened_in = stack.pop()
            if opened_in >= 0:
                segment = next_segment  # the rest of the stem is its own curve from here
                next_segment += 1
            else:
                segment = -1
            yield _NAN_ROW
            yield (pos[0], pos[1], pos[2], diam, segment)

        elif command == "{":
            segment = next_segment
            next_segment += 1
            yield (pos[0], pos[1], pos[2], diam, segment)
        elif command == "}":
            segment = -1

        elif command.isalpha() and command.isupper():
            continue  # a non-terminal left at depth 0 draws nothing
        else:
            raise ValueError(f"unknown command {command!r}")

    if stack:
        raise ValueError(f"{len(stack)} '[' without a matching ']'")


_SHAPED_EVENTS = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
                  "collision_redraws", "collision_terminations", "collision_truncated_stems") + GUIDANCE_EVENTS


def _sample_stem(control, subdivisions):
    """The samples utils.interpolate_segments would produce for these control rows."""
    control = np.asarray(control, dtype=float)
    return bspline(control, subdivisions) if len(control) >= 3 else control.copy()


def _arc_positions(samples):
    steps = np.linalg.norm(np.diff(samples[:, :3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _flush_stem(control, subdivisions, avoid, stem, events):
    """
    Samples a buffered spline stem and, with an avoider, shortens it to its
    longest collision-free prefix of control points, committing the result.

    Returns:
        tuple: (samples, truncated) with samples an (m, 4) array and truncated
        True when at least one control point was dropped.
    """
    control = list(control)
    truncated = False
    while True:
        samples = _sample_stem(control, subdivisions)
        if avoid is None:
            return samples, truncated
        arcs = _arc_positions(samples)
        bad = avoid.conflicts(samples[:, :3], samples[:, 3] / 2.0, stem, arcs)
        if not bad.any() or len(control) == 1:
            break
        control.pop()
        truncated = True
    if truncated:
        events["collision_truncated_stems"] += 1
    avoid.commit(samples[:, :3], samples[:, 3] / 2.0, stem, arcs)
    return samples, truncated


def _interpret_shaped(turtle_program, d0, position, direction, perpendicular, bounds,
                      walk, avoid, events, subdivisions):
    """
    Interpretation with a walk and/or collision avoidance. Yields final polyline
    rows (segment -1): an unshortened spline stem gives exactly the samples the
    default path gives after utils.interpolate_segments, a walked stem gives
    2**subdivisions points per sub-segment.

    Every polyline (a braced stem, a run of bare moves, or the continuation of
    a stem after a branch pinned inside it) is registered with the avoider as a
    "stem" with the stem it descends from as parent, so that the avoider can
    excuse the overlaps at a junction and nothing else.
    """
    pos, heading, perp, diam, low, high = _start_state(d0, position, direction, perpendicular, bounds)
    if events is None:
        events = {}
    for key in _SHAPED_EVENTS:
        events.setdefault(key, 0)
    if subdivisions < 0:
        raise ValueError("subdivisions cannot be negative")
    steps_per_move = 2 ** subdivisions

    stack = []
    depth = 0
    pruned_at = None
    braced = False          # inside { }
    stem = None             # id of the polyline being extended, or None
    last_stem = None        # the stem the next polyline descends from
    control = None          # buffered control rows of a spline stem
    arc = 0.0               # arc position along the current stem
    rule = None             # the guidance rule of the polyline being walked
    rule_due = False        # the rule is chosen at the polyline's first move
    counter = [0]

    def open_stem(parent, junction, junction_diameter):
        if avoid is not None:
            return avoid.open_stem(parent, junction, junction_diameter)
        counter[0] += 1
        return counter[0] - 1

    def start_stem(parent, junction, junction_diameter):
        """Opens a polyline at `junction` and records its first point."""
        new = open_stem(parent, junction, junction_diameter)
        if avoid is not None:
            avoid.commit(junction[None, :], [junction_diameter / 2.0], new, [0.0])
        return new

    yield (pos[0], pos[1], pos[2], diam, -1)

    for command, params in tokenise(turtle_program):
        if pruned_at is not None and command not in ("[", "]"):
            continue

        if command in MOVE_COMMANDS:
            length = params[0] if params else getLength(diam)
            if len(params) > 1 and params[1] > 0.0:
                diam = params[1]

            if braced and walk is not None:
                # the walk: rotate the frame, step, check, redraw or terminate
                h = length / steps_per_move
                std = step_std(h, diam, walk.persistence)
                radius = diam / 2.0
                terminated = False
                if walk.guidance is not None and rule_due:
                    rule = walk.guidance.rule_for(diam)
                    rule_due = False
                for _ in range(steps_per_move):
                    if walk.guidance is not None:
                        heading, perp = walk.guidance.steer(heading, perp, pos, rule, diam, h, arc, events)
                    bound_failures = 0
                    collision_failures = 0
                    while True:
                        # every redraw widens the turn by one more multiple of the step's
                        # own spread, so that a walk heading into an obstacle can steer
                        # away from it instead of repeating almost the same step
                        widening = 1.0 + bound_failures + collision_failures
                        new_heading, new_perp = perpendicular_rotation(walk.rng, heading, perp, std * widening)
                        candidate = pos + h * new_heading
                        if low is not None and (np.any(candidate < low) or np.any(candidate > high)):
                            bound_failures += 1
                            events["walk_bound_redraws"] += 1
                            if bound_failures > walk.bound_attempts:
                                events["walk_bound_terminations"] += 1
                                terminated = True
                                break
                            continue
                        if avoid is not None and avoid.conflicts(candidate[None, :], [radius], stem,
                                                                 [arc + h])[0]:
                            collision_failures += 1
                            events["collision_redraws"] += 1
                            if collision_failures > avoid.attempts:
                                events["collision_terminations"] += 1
                                terminated = True
                                break
                            continue
                        break
                    if terminated:
                        break
                    heading, perp, pos = new_heading, new_perp, candidate
                    arc += h
                    if avoid is not None:
                        avoid.commit(pos[None, :], [radius], stem, [arc])
                    yield (pos[0], pos[1], pos[2], diam, -1)
                if terminated:
                    pruned_at = depth
                continue

            step = pos + length * heading
            if low is not None and (np.any(step < low) or np.any(step > high)):
                # the branch leaves the box: draw what the stem has so far, then drop the rest
                events["bound_terminations"] += 1
                if control is not None:
                    samples, _ = _flush_stem(control, subdivisions, avoid, stem, events)
                    for row in samples:
                        yield (row[0], row[1], row[2], row[3], -1)
                    control = None
                pruned_at = depth
                continue
            if braced:
                pos = step
                control.append((pos[0], pos[1], pos[2], diam))
                continue

            # a bare move outside braces is a polyline vertex, checked as one point
            if stem is None:
                stem = start_stem(last_stem, pos, diam)
                arc = 0.0
            arc += length
            if avoid is not None and avoid.conflicts(step[None, :], [diam / 2.0], stem, [arc])[0]:
                events["collision_terminations"] += 1
                arc -= length
                pruned_at = depth
                continue
            pos = step
            if avoid is not None:
                avoid.commit(pos[None, :], [diam / 2.0], stem, [arc])
            yield (pos[0], pos[1], pos[2], diam, -1)

        elif command in ("+", "-"):
            if braced and walk is not None:
                continue                      # the zig-zag the walk replaces
            sign = 1.0 if command == "+" else -1.0
            heading = rotate_about(heading, perp, sign * _angle(params, lambda: calBifurcation(diam)["th1"]))
        elif command == "/":
            perp = rotate_about(perp, heading, _angle(params, lambda: lg.roll_angle))
        elif command == "*":
            perp = rotate_about(perp, heading, -_angle(params, lambda: lg.roll_angle))

        elif command == "[":
            depth += 1
            if pruned_at is not None:
                continue
            if control is not None:           # a branch inside a stem pins the stem here
                samples, truncated = _flush_stem(control, subdivisions, avoid, stem, events)
                for row in samples:
                    yield (row[0], row[1], row[2], row[3], -1)
                pos = samples[-1, :3].copy()
                diam = float(samples[-1, 3])
                control = None
                if truncated:
                    # the stem ended before the branch point: the branch and the rest
                    # of the stem go with it, up to the bracket that encloses them
                    pruned_at = depth - 1
                    continue
            parent_here = stem if stem is not None else last_stem
            stack.append((pos, heading, perp, diam, braced, stem, parent_here))
            last_stem = parent_here
            stem = None
            braced = False
        elif command == "]":
            depth -= 1
            if pruned_at is not None:
                if depth >= pruned_at:
                    continue
                pruned_at = None
            if not stack:
                raise ValueError("']' without a matching '['")
            pos, heading, perp, diam, braced, stem, last_stem = stack.pop()
            control = None
            yield _NAN_ROW
            if braced:
                # the rest of the stem continues as its own curve from the branch point,
                # whose restored position is that curve's first point
                stem = start_stem(stem, pos, diam)
                arc = 0.0
                rule_due = True
                if walk is None:
                    control = [(pos[0], pos[1], pos[2], diam)]
                else:
                    yield (pos[0], pos[1], pos[2], diam, -1)
            else:
                yield (pos[0], pos[1], pos[2], diam, -1)
                stem = None

        elif command == "{":
            braced = True
            if stem is not None:
                last_stem = stem
            stem = start_stem(last_stem, pos, diam)
            arc = 0.0
            rule_due = True
            if walk is None:
                control = [(pos[0], pos[1], pos[2], diam)]
            else:
                yield (pos[0], pos[1], pos[2], diam, -1)
        elif command == "}":
            if control is not None:
                samples, truncated = _flush_stem(control, subdivisions, avoid, stem, events)
                for row in samples:
                    yield (row[0], row[1], row[2], row[3], -1)
                pos = samples[-1, :3].copy()
                diam = float(samples[-1, 3])
                control = None
                if truncated:
                    pruned_at = depth
            braced = False
            if stem is not None:
                last_stem = stem
            stem = None

        elif command.isalpha() and command.isupper():
            continue
        else:
            raise ValueError(f"unknown command {command!r}")

    if control is not None:                    # a program ending inside a stem
        samples, _ = _flush_stem(control, subdivisions, avoid, stem, events)
        for row in samples:
            yield (row[0], row[1], row[2], row[3], -1)
    if stack:
        raise ValueError(f"{len(stack)} '[' without a matching ']'")
