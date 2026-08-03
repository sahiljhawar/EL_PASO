<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

[![PyPi](https://badge.fury.io/py/el-paso.svg)](https://badge.fury.io/py/el-paso)
[![Python version](https://img.shields.io/pypi/pyversions/el-paso.svg)](https://badge.fury.io/py/el-paso)
[![Tests](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml/badge.svg)](https://github.com/GFZ/EL_PASO/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/GFZ/EL_PASO/badge.svg?branch=main)](https://coveralls.io/github/GFZ/EL_PASO?branch=main)
[![Docs](https://app.readthedocs.org/projects/el-paso/badge/?version=latest)](https://el-paso.readthedocs.io/en/latest/)
[![REUSE status](https://api.reuse.software/badge/github.com/GFZ/EL_PASO)](https://api.reuse.software/info/github.com/GFZ/EL_PASO)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GFZ/EL_PASO/executable_paper?urlpath=%2Fdoc%2Ftree%2Fpaper%2Fel_paso_executable_paper.ipynb)

# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`EL-PASO` is a Python framework designed to streamline the download, processing, and saving of satellite particle observation data.

Its primary purpose is to prepare and standardize particle data for use in radiation belt modeling.

## Features

- **Format Flexibility:** Capable of handling different input formats including `cdf`, `netcdf`, `h5`, `ascii`, and `json`
- **Integrated Processing:** Provides a comprehensive set of functions for common particle data analysis tasks
- **Supports Metadata:** Stores all processing and metadata alongside the data, ensuring full traceability and reproducibility.
- **Standardized output files:** Saving processed data in different standards (e.g. PRBEM) to enable easy loading and sharing of processed data

Full documentation can be viewed [here](https://el-paso.readthedocs.io/en/latest/).

## Available processing scripts

- **Arase**
    - MEPe
    - XEP real-time
    - PWE density
- **GOES-R**
    - MPS-High real-time
    - MPS-High
- **ESA**
    - NGRM satellites
- **POES**
    - MEPED (electrons)
    - TED (electrons)
- **PROBA-V**
    - EPT (electrons and protons)
- **Van Allen Probes**
    - HOPE (electrons) and ECT-combined
    - EMFISIS and EFW density

## Installation Guide



# Installation

### Step 1: Clone the Repository

Begin by cloning the EL-PASO repository and navigating into its directory.

```bash
git clone https://github.com/GFZ/EL_PASO.git
cd EL_PASO
```

### Step 2: Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install .
```

The custom `setup.py` script will automatically download and compile the IRBEM Fortran library during this step.

### Verifying the Installation

You can validate your installation by running the minimal example located in *examples*:

```bash
python examples/minimal_example.py
```

## Citation

If you use `EL-PASO` in your research, please cite the associated preprint:

> Haas, B., Drozdov, A. Y., and Jhawar, S. 	EL-PASO: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space. ESS Open Archive. https://essopenarchive.org/doi/full/10.22541/essoar.15002644/v1

```bibtex
@article{
doi:10.22541/essoar.15002644/v1,
author = {Bernhard Haas  and Alexander Y. Drozdov  and Sahil Jhawar },
title = {EL-PASO: An Open-Source Python Library for Processing and Standardizing Particle Measurements Taken in Space},
journal = {ESS Open Archive},
volume = {2026},
number = {0502},
pages = {},
year = {2026},
doi = {10.22541/essoar.15002644/v1},
URL = {https://essopenarchive.org/doi/abs/10.22541/essoar.15002644/v1},
eprint = {https://essopenarchive.org/doi/pdf/10.22541/essoar.15002644/v1}}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgements

This work has been funded by the German Research Foundation (NFDI4Earth, DFG project no. 460036893, https://www.nfdi4earth.de/).
The authors acknowledge the work of Mátyás Szabó-Roberts who led the foundation for the EL-PASO framework.

The thank the authors of the [IRBEM library](https://github.com/PRBEM/IRBEM) for providing their code.
