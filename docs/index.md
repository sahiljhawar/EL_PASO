<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache 2.0
-->

# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`el_paso` is a Python framework designed to streamline the download, processing, and saving of satellite particle observation data.

Its primary purpose is to prepare and standardize particle data for use in radiation belt modeling.

This work has been funded by the German Research Foundation (NFDI4Earth, DFG project no. 460036893, https://www.nfdi4earth.de/).

## Features

- Capable of handling different input formats (cdf, netcdf, h5, ascii, json)
- Processing functions most commonly used for analyzing particle measurements are available
- Metadata associating with the processing are storred alongside data
- Saving processed data in different standards (e.g. PRBEM) to enable easy loading of processed data

## Examples

Examples can be found in the *examples* folder and include processing scripts for Van Allen Probes, Arase, and POES.