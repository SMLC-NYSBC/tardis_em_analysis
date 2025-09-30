from tardis_em_analysis._version import version
import os

# Temporal fallback for mps devices
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__version__ = version
