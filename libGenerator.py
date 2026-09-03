"""
Bifurcation and segment-length calculations for the V-System grammars.

The functions follow the libGenerator class of Galarreta-Valverde (2012,
Appendix B). Daughter diameters obey Murray's law with exponent k, the first
daughter diameter is drawn from a Gaussian about the symmetric optimum, and
the bifurcation angles follow Zamir's minimum-lumen-volume rule (Zamir 2001;
Liu et al. 2010) expressed in the asymmetry ratio alpha = d2 / d1.

Parameters are module-level values set with setProperties(); any key left
out of the dictionary keeps its default.
"""
import math
import numpy as np

default = {
    "k": 3,                  # Murray's law exponent
    "epsilon": 10.0,         # length-to-diameter ratio of a segment (propCD in the source)
    "randmarg": 0.2,         # relative half-width of the segment-length distribution
    "sigma": 5,              # d_opt / sigma is the standard deviation of the first daughter diameter
    "stochparams": True,     # draw the first daughter diameter stochastically
    "roll_angle": 70.0,      # roll of the perpendicular vector after each daughter turn, degrees
    "stem_angle": 25.0,      # turn between the sub-segments of a stem, degrees
    "aneurysm_prob": 0.02,   # per sub-segment probability of a local dilation
    "aneurysm_factor": 1.5,  # diameter multiplier inside a dilation
    "stenosis_prob": 0.02,   # per sub-segment probability of a local constriction
    "stenosis_factor": 0.5,  # diameter multiplier inside a constriction
}


def setProperties(properties=None):
    """
    Sets the module-level parameters.

    Args:
        properties (dict or None): values to override; keys missing from the
            dictionary take their default. None resets everything to defaults.

    Raises:
        ValueError: on an unknown key or an out-of-range value.
    """
    global k, epsilon, randmarg, sigma, stochparams
    global roll_angle, stem_angle, aneurysm_prob, aneurysm_factor, stenosis_prob, stenosis_factor

    merged = dict(default)
    if properties is not None:
        unknown = sorted(set(properties) - set(default))
        if unknown:
            raise ValueError(f"unknown properties {unknown}; valid keys are {sorted(default)}")
        merged.update(properties)

    if not 0.0 <= merged["randmarg"] < 1.0:
        raise ValueError(
            f"randmarg must be a fraction in [0, 1), got {merged['randmarg']!r}: it is the relative "
            "half-width of the segment-length distribution, not an absolute margin")
    for key in ("k", "epsilon", "sigma", "aneurysm_factor", "stenosis_factor"):
        if not merged[key] > 0:
            raise ValueError(f"{key} must be positive, got {merged[key]!r}")
    for key in ("aneurysm_prob", "stenosis_prob"):
        if not 0.0 <= merged[key] <= 1.0:
            raise ValueError(f"{key} must be a probability, got {merged[key]!r}")
    if merged["aneurysm_prob"] + merged["stenosis_prob"] > 1.0:
        raise ValueError("aneurysm_prob + stenosis_prob must not exceed 1")

    k = merged["k"]
    epsilon = merged["epsilon"]
    randmarg = merged["randmarg"]
    sigma = merged["sigma"]
    stochparams = merged["stochparams"]
    roll_angle = merged["roll_angle"]
    stem_angle = merged["stem_angle"]
    aneurysm_prob = merged["aneurysm_prob"]
    aneurysm_factor = merged["aneurysm_factor"]
    stenosis_prob = merged["stenosis_prob"]
    stenosis_factor = merged["stenosis_factor"]


def calBifurcation(d0):
    """
    Calculates the daughter diameters and bifurcation angles for a parent diameter.

    Args:
        d0 (float): parent diameter, positive.

    Returns:
        dict: d0, d1, d2 (diameters), th1, th2 (angles in degrees) and co
        (a segment length for d0).
    """
    if not d0 > 0:
        raise ValueError(f"bifurcation requires a positive parent diameter, got {d0!r}")

    dOpti = d0 / 2 ** (1.0 / k)
    if stochparams:
        d1 = abs(np.random.normal(dOpti, dOpti / sigma))
    else:
        d1 = dOpti
    if d1 >= d0:
        d1 = dOpti  # the source resets an impossible draw to the symmetric optimum

    d2 = (d0 ** k - d1 ** k) ** (1.0 / k)
    alpha = d2 / d1

    # Zamir's minimum-lumen-volume angles in terms of the asymmetry ratio
    xtmp = (1 + alpha ** 3) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha ** 3) ** (2.0 / 3))
    a1 = math.acos(xtmp / xtmpb)

    xtmp = (1 + alpha ** 3) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha ** 3) ** (2.0 / 3))
    a2 = math.acos(xtmp / xtmpb)

    return {
        "d1": d1,
        "d2": d2,
        "d0": d0,
        "th1": a1 * 180 / math.pi,
        "th2": a2 * 180 / math.pi,
        "co": getLength(d0),
    }


def getLength(d0):
    """
    Returns a segment length for a parent diameter: epsilon * d0 scaled by a
    uniform factor in [1 - randmarg, 1 + randmarg]. Because the margin is
    relative, the length stays positive and proportional to the diameter at
    every generation.
    """
    c0 = d0 * epsilon
    return c0 * np.random.uniform(1.0 - randmarg, 1.0 + randmarg)


setProperties()
