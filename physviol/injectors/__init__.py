"""Injector registry. Importing this module registers every injector."""
from .base import (Injector, InterventionPlan, available, get,  # noqa: F401
                   register)
from . import (contact, dynamics, equilibrium, identity,  # noqa: F401
               kinematics, optical)
