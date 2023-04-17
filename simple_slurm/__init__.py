from .__about__ import __version__

from .core import Slurm
from .hyak import parse_hyakalloc, find_best_allocation, Resources

# create a dummy Slurm object, this forces the creation of attributes for
# file patterns and output environment variables
_ = Slurm()

__all__ = [
    '__version__',
]
