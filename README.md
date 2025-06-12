# ELobarative Particle Analysis from Satellite Observations (EL PASO)

## Installation 

Start by cloning the repository:
```
git clone git@rbm9.gfz-potsdam.de:bhaas/EL-PASO.git --recursive
```

By cloning it recursively, you will also clone the IRBEM and data_management repositories.
Next, create a virtual environment and activate it:
```
cd EL-PASO
python3 -m venv venv
source venv/bin/activate
```
Now we can start installing EL-PASO using pip:
```
pip install .
```
and also install data_management:
```
pip install data_management/
```
For IRBEM, we have to compile the library first. Navigate to the IRBEM folder and call the fortran compiler:
```
cd IRBEM
make OS=linux64 ENV=gfortran64 all
make OS=linux64 ENV=gfortran64 install
pip install python/
```
This compiles the *libirbem.so* file, puts it in the IRBEM folder and installs the python wrapper for IRBEM.