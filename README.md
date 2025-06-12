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
Afterwards, we can install data_management:
```
pip install data_management/
```

For IRBEM, we have to compile the library first. Navigate to the IRBEM folder and call the fortran compiler:
```
cd IRBEM
make OS=linux64 ENV=gfortran64 all
make OS=linux64 ENV=gfortran64 install
```
This compiles the *libirbem.so* file and puts it in the IRBEM folder.
We apply a small patch to the IRBEM python wrapper, which adds the Lstar calculation, and install the wrapper afterwards:
```
cp ../IRBEM.py python/IRBEM/
pip install python/
```