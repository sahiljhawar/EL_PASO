# ELaborative Particle Analysis from Satellite Observations (EL-PASO)

`el_paso` is a framework for downloading, processing, and saving particle observations from satellite missions.
Its main goal is providing a framework for saving satellite data in a standardized way useful for radiation belt modelling.

It was developed as an Incubator Project funded by NFDI4Earth.

## Features

- Capable of handling different input formats (cdf, netcdf, h5, ascii, json)
- Processing functions most commonly used for analyzing particle measurements are available
- Metadata associating with the processing are storred alongside data
- Saving processed data in different standards (e.g. PRBEM) to enable easy loading of processed data

## Examples

Examples can be found in the *examples* folder and include processing scripts for Van Allen Probes, Arase, and POES.