# Getting Started

## Installation

## Classes

EL-PASO heavily relies on the usage of three classes: [Variable](../API_reference/variable.md), [SourceFile](../API_reference/source_file.md), and [SaveStandard](../API_reference/save_standard.md).

The *Variable* class is a container holding data and associated Metadata info like units and processing notes. All data which should be used for processing or saving should be put wrapped by a Variable class. Its usage is similar to variables in a cdf file.

The *SourceFile* class is responsible for downloading, loading, and extracting data and returning them as Variables.

The *SaveStandard* class is a base class for classes which describe how the processed data will be saved. This includes how the variables should be split up in different files and with which unit the variables should be saved.

Implementing a processing pipeline using EL-PASO means to describe the source files, process the Variables, and specify a save standard.

## Standards

The strength of EL-PASO is that it is easy to enforce a certain data standard. The data standard describes the dimensions, units, and names of the data set. For example, the standard describes that a measurement of the unidirectional differential electron flux has always dimensions of (Time, energy, pitch angle), a unit of 1/(keV str s cm^2), and is called *FEDU*. The standards are implemented as a SQL-database.

A Variable can be associated with a given standard. By letting the framework know that a certain variable holds a flux measurement, the framework will ensure that the data has the correct dimensions and units, and is also searchable through its name. We will see later what this means in practise.

## Running the minimal example