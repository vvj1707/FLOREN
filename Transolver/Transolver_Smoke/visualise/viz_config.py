from pathlib import Path

# Project structure
THIS_FILE = Path(__file__).resolve()
VIS_DIR = THIS_FILE.parent
FNO_DIR = VIS_DIR.parent
PROJECT_DIR = FNO_DIR.parent

# Inputs
DATA_DIR = PROJECT_DIR / "data" / "warped-ifw"
RESULTS_DIR = FNO_DIR / "results"

# Outputs
OUT_DIR = VIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Split settings (must match training)
TEST_FRAC = 0.15
SEED = 42

# Plot styling
FIG_DPI = 140
ANIM_DPI = 130
ANIM_FPS = 2

CMAP_MAIN = "turbo"       # good CFD-like palette
CMAP_ERR = "magma"
CMAP_PRESS = "coolwarm"

# 2D section settings
SECTION_AXIS = "y"        # "x" | "y" | "z"
SECTION_QUANTILE = 0.5
SECTION_BAND_FRAC = 0.015  # relative to axis range
GRID_NX = 300
GRID_NY = 180
GAUSS_SIGMA = 1.0          # set 0.0 to disable smoothing