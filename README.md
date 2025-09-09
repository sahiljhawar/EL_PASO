<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences

SPDX-License-Identifier: Apache 2.0
-->

![REUSE Compliance](https://img.shields.io/reuse/compliance/:remote)

# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`EL-PASO` is a Python framework designed to streamline the download, processing, and saving of satellite particle observation data.

Its primary purpose is to prepare and standardize particle data for use in radiation belt modeling.

## Features

- **Format Flexibility:** Capable of handling different input formats including `cdf`, `netcdf`, `h5`, `ascii`, and `json`
- **Integrated Processing:** Provides a comprehensive set of functions for common particle data analysis tasks
- **Supports Metadata:** Stores all processing and metadata alongside the data, ensuring full traceability and reproducibility.
- **Standardized output files:** Saving processed data in different standards (e.g. PRBEM) to enable easy loading and sharing of processed data

## Installation Guide

### Step 1: Clone the Repository

Begin by cloning the EL-PASO repository and navigating into its directory.

```bash
git clone https://github.com/GFZ/EL_PASO.git
cd el_paso
```

### Step 2: Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install the EL PASO Package

Install the core EL-PASO package using pip.

```bash
pip install .
```

### Step 4: Prepare and Install the IRBEM Library

The [IRBEM library](https://github.com/PRBEM/IRBEM) is the backend for magnetic field calculations. It is already included as a submodule and can be cloned by calling:

```bash
git submodule init
git submodule update
```

Now, the IRBEM repository has been cloned to the *IRBEM* folder. Next, we have to compile the IRBEM library. You can follow their compilation instructions at on [github](https://github.com/PRBEM/IRBEM) but on Linux, you should be able to simple call:

```bash
cd IRBEM
make
make install .
```

These commands compile the *libirbem.so* file and copy it to the root of the IRBEM directory.

Before installing the python bindings, we have to apply a custom patch to the python wrapper, since some necessary functionalities are missing in the official IRBEM python wrapper. For this, copy the IRBEM.py file from the root of the EL-PASO repository to the IRBEM python wrapper:

```bash
cp ../IRBEM.py python/IRBEM
```

Now we can install the wrapper:
```bash
pip install python/
```
### Verifying the Installation

You can validate your installation by running the minimal example located in *examples*:

```bash
python3 examples/minimal_example.py
```

## Viewing the documentation

EL-PASO uses `mkdocs` for building its documentation. To view it locally in your browser, run the following command from the root of the repository:

```bash
mkdocs serve
```

You can then access the documentation at `http://127.0.0.1:8000/`.

## Acknowledgements

EL-PASO was initially developed as an Incubator Project funded by NFDI4Earth in 2025.

The authors acknowledge the work of Mátyás Szabó-Roberts who led the foundation for the EL-PASO framework.
