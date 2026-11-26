"""Tone-mapping constants (moved verbatim from NHC.py)."""


# Constants
DEFAULT_TONE_MAP = "perceptual"


DEFAULT_PREGAMMA = "1.0"


DEFAULT_AUTOEXPOSURE = "1.0"


# The last line holds earlier names, still accepted (see renderings.canonical).
SUPPORTED_TONE_MAPS = {"faithful", "natural", "punchy", "golden",
                       "perceptual", "reinhard", "hable", "aces", "drago03", "mantiuk06", "adaptive", "neural",
                       "agx", "agx_punchy", "agx_golden", "filmic", "uncharted2"}

# What SDR white stands for when nothing better is known: the reference white of HDR
# programmes (ITU-R BT.2408). The application passes the screen's own SDR white instead.
REFERENCE_WHITE_NITS = 203.0


# The rendering a caller gets when it names none.
DEFAULT_RENDER_METHOD = "faithful"

# The engine's working space is scRGB x 0.1, the scale the JXR loader applies
# (display peak 1000 / HDR peak 10000). scRGB 1.0 is 80 nits, so 1.0 here is 800 nits.
ENGINE_UNIT_NITS = 800.0
