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
"""
import math
import re

import numpy as np

import libGenerator as lg
from libGenerator import calBifurcation, getLength
from utils import rotate_about, unit

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
                               perpendicular=(0.0, 0.0, 1.0)):
    """
    Interprets a turtle program and yields its points.

    Args:
        turtle_program (str): the L-system string.
        d0 (float): initial diameter.
        position, direction, perpendicular: initial state; direction and
            perpendicular must be orthogonal. The defaults are the source's.

    Yields:
        tuple: (x, y, z, diameter, segment). `segment` is the index of the
        enclosing {...} segment, or -1 outside braces. The initial position
        is yielded first. A row of NaN marks the end of a branch (a ']'),
        after which the restored point is yielded again -- with a fresh
        segment index if the branch was opened inside a segment, so that the
        continuation is interpolated as its own curve from the branch point.

    Raises:
        ValueError: on a malformed program or unbalanced brackets.
    """
    pos = np.asarray(position, dtype=float)
    heading = unit(direction)
    perp = unit(perpendicular)
    if abs(heading @ perp) > 1e-9:
        raise ValueError("direction and perpendicular must be orthogonal")
    diam = float(d0)
    if not diam > 0:
        raise ValueError(f"the initial diameter must be positive, got {d0!r}")

    stack = []
    segment = -1
    next_segment = 0
    yield (pos[0], pos[1], pos[2], diam, segment)

    for command, params in tokenise(turtle_program):
        if command in MOVE_COMMANDS:
            length = params[0] if params else getLength(diam)
            if len(params) > 1 and params[1] > 0.0:
                diam = params[1]
            pos = pos + length * heading
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
            stack.append((pos, heading, perp, diam, segment))
            segment = -1  # a branch leaving an open stem pins the stem here
        elif command == "]":
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
