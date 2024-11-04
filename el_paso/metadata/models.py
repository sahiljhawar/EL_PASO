from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class ConfigVariables(Base):
    __tablename__ = 'ConfigVariables'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    mission = Column(String)
    instrument = Column(String)
    description = Column(String)
    config_variables = relationship("ConfigVariable", back_populates="config")


class ConfigVariable(Base):
    __tablename__ = 'ConfigVariable'
    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey('ConfigVariables.id'))
    file_location = Column(String)
    workspace_name = Column(String, nullable=False)
    standard_name = Column(String, nullable=False)
    file_unit = Column(String)
    unit_conversion = Column(String)
    binning_method = Column(String)
    fill_dimensions = Column(String)
    config_notes = Column(String)
    config = relationship("ConfigVariables", back_populates="config_variables")


class StandardsTable(Base):
    __tablename__ = 'StandardsTable'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    definition = Column(String)
    function_name = Column(String)
    standard_variables = relationship("StandardVariable", back_populates="standard")


class StandardVariable(Base):
    __tablename__ = 'StandardVariable'
    id = Column(Integer, primary_key=True)
    standard_id = Column(Integer, ForeignKey('StandardsTable.id'))
    variable_type = Column(String, nullable=False)
    standard_name = Column(String, nullable=False)
    standard_description = Column(String)
    standard_notes = Column(String)
    standard_unit = Column(String)
    standard = relationship("StandardsTable", back_populates="standard_variables")
    save_variables = relationship("SaveVariable", back_populates="standard_variable")
    species_variables = relationship("SpeciesVariables", back_populates="standard_variable")
    mfms_variables = relationship("MFMsVariables", back_populates="standard_variable")


class SaveStandards(Base):
    __tablename__ = 'SaveStandards'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    function_name = Column(String)
    folder_template = Column(String)
    file_templates = Column(String)  # This could be JSON encoded to store multiple templates
    extra_information = Column(String)
    save_variables = relationship("SaveVariable", back_populates="save_standard")


class SaveVariable(Base):
    __tablename__ = 'SaveVariable'
    id = Column(Integer, primary_key=True)
    variable_type = Column(String, nullable=False)
    save_standard_id = Column(Integer, ForeignKey('SaveStandards.id'))
    standardVariable_id = Column(Integer, ForeignKey('StandardVariable.id'))
    standard_name = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    unit_conversion = Column(String)
    file_unit = Column(String)
    file_notes = Column(String)
    save_standard = relationship("SaveStandards", back_populates="save_variables")
    standard_variable = relationship("StandardVariable", back_populates="save_variables")
    species_save_variables = relationship("SpeciesSaveVariables", back_populates="save_variable")
    mfms_save_variables = relationship("MFMsSaveVariables", back_populates="save_variable")


class MagFieldModels(Base):
    __tablename__ = 'MagFieldModels'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    short_name = Column(String, nullable=False)
    long_name = Column(String, nullable=False)
    irbem_internal = Column(Integer)
    irbem_external = Column(Integer)
    default_options = Column(String)
    standard_tag = Column(String, nullable=False)
    description = Column(String)
    mfms_variables = relationship("MFMsVariables", back_populates="mfm")
    mfms_save_variables = relationship("MFMsSaveVariables", back_populates="mfm")


class Epochs(Base):
    __tablename__ = 'Epochs'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    epoch_name = Column(String, nullable=False)
    description = Column(String)


class Species(Base):
    __tablename__ = 'Species'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    flux_name = Column(String, nullable=False)
    rest_mass_mev = Column(Float)
    description = Column(String)
    species_variables = relationship("SpeciesVariables", back_populates="species")
    species_save_variables = relationship("SpeciesSaveVariables", back_populates="species")


class CoordinateSystems(Base):
    __tablename__ = 'CoordinateSystems'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    definition = Column(String, nullable=False)
    abbreviation = Column(String)
    standard_unit = Column(String)


class StandardVariableTypes(Base):
    __tablename__ = 'StandardVariableTypes'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    definition = Column(String)


class SpeciesVariables(Base):
    __tablename__ = 'SpeciesVariables'
    standard_variable_id = Column(Integer, ForeignKey('StandardVariable.id'), primary_key=True)
    species_id = Column(Integer, ForeignKey('Species.id'), primary_key=True)
    description = Column(String)
    standard_variable = relationship("StandardVariable", back_populates="species_variables")
    species = relationship("Species", back_populates="species_variables")


class SpeciesSaveVariables(Base):
    __tablename__ = 'SpeciesSaveVariables'
    save_variable_id = Column(Integer, ForeignKey('SaveVariable.id'), primary_key=True)
    species_id = Column(Integer, ForeignKey('Species.id'), primary_key=True)
    description = Column(String)
    save_variable = relationship("SaveVariable", back_populates="species_save_variables")
    species = relationship("Species", back_populates="species_save_variables")


class MFMsVariables(Base):
    __tablename__ = 'MFMsVariables'
    standard_variable_id = Column(Integer, ForeignKey('StandardVariable.id'), primary_key=True)
    mfm_id = Column(Integer, ForeignKey('MagFieldModels.id'), primary_key=True)
    description = Column(String)
    standard_variable = relationship("StandardVariable", back_populates="mfms_variables")
    mfm = relationship("MagFieldModels", back_populates="mfms_variables")


class MFMsSaveVariables(Base):
    __tablename__ = 'MFMsSaveVariables'
    save_variable_id = Column(Integer, ForeignKey('SaveVariable.id'), primary_key=True)
    mfm_id = Column(Integer, ForeignKey('MagFieldModels.id'), primary_key=True)
    description = Column(String)
    save_variable = relationship("SaveVariable", back_populates="mfms_save_variables")
    mfm = relationship("MagFieldModels", back_populates="mfms_save_variables")


class VariableDependencies(Base):
    __tablename__ = 'VariableDependencies'
    source_variable_id = Column(Integer, ForeignKey('StandardVariable.id'), primary_key=True)
    dependency_id = Column(Integer, ForeignKey('StandardVariable.id'), primary_key=True)
    description = Column(String)
