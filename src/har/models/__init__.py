"""HAR model architectures.

Public API:
- HARModel: abstract base class for all sequence models
- build_model: factory function to instantiate a model from config
- register_arch: decorator to register new architectures
"""

from har.models.base import HARModel, build_model, register_arch

# Import architecture modules to trigger @register_arch decorators
import har.models.tcn  # noqa: F401
import har.models.bigru  # noqa: F401
import har.models.transformer  # noqa: F401
import har.models.bigru  # noqa: F401

__all__ = ["HARModel", "build_model", "register_arch"]
