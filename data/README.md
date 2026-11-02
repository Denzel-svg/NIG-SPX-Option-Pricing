# Data directory

- `raw/`: original source files. Keep them out of version control unless the team agrees otherwise.
- `cleaned/`: cleaned and validated datasets used by the notebooks and calibration code.

The calibration workflow should consume the long-format calibration candidate dataset. Do not use a wide strike-by-expiry matrix as the primary calibration input.
