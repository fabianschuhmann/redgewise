"""redgewise package."""

__version__ = "0.1.0"

try:
    from redgewise import suite as suite
except Exception:  # pragma: no cover - keep package import robust during partial installs.
    suite = None

__all__ = ["__version__", "suite"]
