"""E-means clustering, compatible with scikit-learn."""

# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import PackageNotFoundError, version

from sklearn_emeans._emeans import EMeans

try:
    __version__ = version("sklearn-contrib-emeans")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["EMeans", "__version__"]
