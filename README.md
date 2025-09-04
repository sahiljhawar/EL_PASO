# ELaborative Particle Analysis from Satellite Observations (EL PASO)

## Installation

### Installing EL-PASO

After cloning the repository, the main package can be installed using a virtual environment and pip. Make sure your current directory is set to the EL-PASO repository:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

### Preparing and installing IRBEM

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
### Validation of installation

You can validate your installation by running the minimal example located in *examples*:

```bash
python3 examples/minimal_example.py
```

## Viewing the documentation

EL-PASO uses `mkdocs` for building its documentation. If you have EL-PASO installed, you can see the documentation by running

```bash
mkdocs serve
```

and viewing it in your browser under the address http://127.0.0.1:8000/.