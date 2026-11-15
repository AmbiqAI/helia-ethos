# Working in helia-ethos

Shared Ethos-U tooling for Vela compilation, model inspection, fencing, PMU
programs and catalog-backed timing policies. Do not import helia_estimator.
Consumers own execution, runtime integration, energy estimation and results.

Use typed public boundaries with docstrings, and preserve optional extras:
core imports must work without LiteRT or Vela. Hardware facts come from
helia-platforms; compiler and timing assumptions remain explicit model policy.

Run `uv run --frozen python scripts/gates.py` and
`uv run --frozen python scripts/check_distribution.py` before shipping.
Use conventional commit PR titles. Release Please owns versions and CHANGELOG;
do not edit them by hand. Keep generated models and experiment outputs untracked.
