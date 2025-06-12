from .custom_units import *

# useful custom IRBEM aliases
IRBEM_SYSAXIS_GDZ = 0
IRBEM_SYSAXIS_GEO = 1
IRBEM_SYSAXIS_GSM = 2
IRBEM_SYSAXIS_GSE = 3
IRBEM_SYSAXIS_SM  = 4
IRBEM_SYSAXIS_GEI = 5
IRBEM_SYSAXIS_MAG = 6


# import physics
# import utils
from .classes import TimeBinMethod, Variable, SavingStrategy
from . import processing, saving_strategies
from .save import save
from .download import download
from .extract_variables_from_files import extract_variables_from_files, ExtractionInfo