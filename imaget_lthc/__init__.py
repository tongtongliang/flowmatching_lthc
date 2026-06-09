"""Backward-compatible import alias for the renamed package.

Use ``flowmatching_lthc`` in new code. This package remains so older scripts
that import ``imaget_lthc`` keep working.
"""

from flowmatching_lthc import *  # noqa: F401,F403
