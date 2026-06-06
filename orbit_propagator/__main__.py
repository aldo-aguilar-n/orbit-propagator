"""Allow running the package with ``python -m orbit_propagator``."""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
