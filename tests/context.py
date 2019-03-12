import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import raytracer

# Note: this must be imported at the beginning of each test file
# and allows running the tests without installing the package

# e.g.:
#
#    from .context import raytracer
