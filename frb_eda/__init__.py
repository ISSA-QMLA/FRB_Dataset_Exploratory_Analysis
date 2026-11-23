"""Shared code for the FRB pilot EDA notebooks in Data/.

The notebooks carry the reasoning; this package carries the functions they share
and the parallel workers for the heavy passes. See config.py for the environment
variables that control paths and worker count.
"""
from . import axes, config, parallel, physics, provenance, sigproc  # noqa: F401
