import json
import sqlite3
import time
from el_paso.metadata.models import Species, MagFieldModels, Epochs, StandardsTable, StandardVariable, \
    VariableDependencies, SaveStandards, SaveVariable, MFMsVariables, SpeciesVariables
from el_paso.metadata.function_import import config


def add_species(active_session, name, flux_name, rest_mass_mev, description):
    new_species = Species(name=name, flux_name=flux_name, rest_mass_mev=rest_mass_mev, description=description)
    species_exists = active_session.query(Species).filter_by(name=name, flux_name=flux_name).first()
    if not species_exists:
        active_session.add(new_species)
    active_session.commit()

    base_strings = generate_base_strings(flux_name)
    mag_field_ids = active_session.query(MagFieldModels).all()
    epochs_list = active_session.query(Epochs).all()
    for base_string in base_strings:
        print('Adding variables for ' + base_string + '...')
        start_time = time.perf_counter()
        create_standard_variables_for_base_string(active_session, base_string, name, mag_field_ids, epochs_list)
        end_time = time.perf_counter()
        print(f"Added variables for {base_string}. Elapsed time is {end_time - start_time} seconds.")


def generate_base_strings(flux_name):
    prefixes = ["F", "EF"]
    types = ["I", "D"]
    directions = ["U", "O", "S"]
    base_strings = []

    for prefix in prefixes:
        for integral_type in types:
            for direction in directions:
                base_strings.append(f"{prefix}{flux_name}{integral_type}{direction}")

    return base_strings


def construct_flux_description(base_string, species_name):
    flux_type = "flux" if base_string.startswith("F") else "energy flux"
    if base_string[-1] == "O":
        direction = "omnidirectional"
    elif base_string[-1] == "U":
        direction = "unidirectional"
    elif base_string[-1] == "S":
        direction = "spin-averaged"
    else:
        direction = ""

    if base_string[-2] == "I":
        measurement = "integral in energy"
    elif base_string[-2] == "D":
        measurement = "differential in energy"
    else:
        measurement = ""

    is_error = "_err" in base_string
    error_prefix = "Uncertainty of " if is_error else ""

    if not is_error:
        direction_capitalized = direction.capitalize()
    else:
        direction_capitalized = direction

    description = f"{error_prefix}{direction_capitalized} {species_name} {flux_type}, {measurement}."

    return description


def add_standard_variable(in_session, variable_data, standard_id, mfm_id, species_id, related_flux, dependency_ids,
                          dependency_names):
    standard_name, standard_description, variable_type, standard_unit, standard_notes = variable_data

    new_variable = StandardVariable(
        standard_name=standard_name,
        standard_description=standard_description,
        standard_unit=standard_unit,
        standard_notes=standard_notes,
        variable_type=variable_type,
        standard_id=standard_id
    )

    if mfm_id:
        new_variable.mfms_variables.append(MFMsVariables(mfm_id=mfm_id))

    if species_id:
        new_variable.species_variables.append(SpeciesVariables(species_id=species_id))

    in_session.add(new_variable)
    in_session.commit()
    in_session.flush()

    if related_flux:
        related_flux_variable = in_session.query(StandardVariable).filter_by(standard_name=related_flux).first()
        dependency_ids.append(related_flux_variable.id)
        dependency_names.append('related flux')

    for index, dependency_id in enumerate(dependency_ids):
        description = dependency_names[index]
        dependency_relation = VariableDependencies(source_variable_id=new_variable.id,
                                                   dependency_id=dependency_id,
                                                   description=description)
        in_session.add(dependency_relation)

    in_session.commit()

    save_standards = in_session.query(SaveStandards).all()
    for save_standard in save_standards:
        add_save_variable(in_session, variable_data, save_standard.id, new_variable.id)


def add_save_variable(in_session, variable_data, save_standard_id, standard_variable_id):
    standard_name, standard_description, variable_type, standard_unit, standard_notes = variable_data

    save_standard = in_session.query(SaveStandards).filter_by(id=save_standard_id).first()

    standard_function = config.get_function(save_standard.function_name)
    file_name, unit_conversion, file_unit, file_notes = standard_function(standard_name, variable_type)
    if not file_unit:
        file_unit = standard_unit

    new_save_variable = SaveVariable(
        variable_type=variable_type,
        standard_name=standard_name,
        file_name=file_name,
        unit_conversion=unit_conversion,
        file_unit=file_unit,
        file_notes=file_notes,
        save_standard_id=save_standard_id,
        standardVariable_id=standard_variable_id
    )

    in_session.add(new_save_variable)
    in_session.commit()
    in_session.flush()

    in_session.commit()


def add_variable_dependency(in_session, source_var_id, dependent_var_id, description="depends on"):
    dependency = VariableDependencies(
        source_variable_id=source_var_id,
        dependency_id=dependent_var_id,
        description=description
    )
    in_session.add(dependency)


def link_variable_dependencies(in_session, base_string):
    epoch_vars = in_session.query(StandardVariable).filter(StandardVariable.standard_name.like(f"Epoch_{base_string}%"),
                                                           StandardVariable.variable_type == "Epoch").all()

    energy_var = in_session.query(StandardVariable).filter_by(standard_name=f"Energy_{base_string}",
                                                              variable_type="Energy").first()

    flux_vars = in_session.query(StandardVariable).filter(StandardVariable.standard_name.like(f"{base_string}%"),
                                                          StandardVariable.variable_type == "Flux").all()

    for flux_var in flux_vars:
        if energy_var:
            add_variable_dependency(in_session, flux_var.id, energy_var.id)
        for epoch_var in epoch_vars:
            add_variable_dependency(in_session, flux_var.id, epoch_var.id)

        if base_string.endswith("U"):
            pa_local_var = in_session.query(StandardVariable).filter_by(standard_name=f"PA_local_OBS_{base_string}",
                                                                        variable_type="PitchAngle").first()
            if pa_local_var:
                add_variable_dependency(in_session, flux_var.id, pa_local_var.id)

    in_session.commit()


def export_flux_variables_to_json(in_database_path):
    conn = sqlite3.connect(in_database_path)
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM StandardVariables WHERE name LIKE '%Flux%'")
    flux_info = cur.fetchone()
    flux_id, flux_name = flux_info

    cur.execute("SELECT * FROM standard_variable WHERE standard_id = ?", (flux_id,))
    flux_variables = cur.fetchall()

    data = {
        "header": {
            "table_name": "StandardVariables",
            "flux_name": flux_name
        },
        "rows": flux_variables
    }

    with open('Flux_variables.json', 'w') as f:
        json.dump(data, f, indent=4)

    conn.close()
    print("Exported flux variables to Flux_variables.json successfully.")


def update_database_from_json(in_database_path, in_json_file_path):
    conn = sqlite3.connect(in_database_path)
    cur = conn.cursor()

    with open(in_json_file_path, 'r') as f:
        data = json.load(f)

    table_name = data['header']['table_name']
    flux_name = data['header']['flux_name']

    cur.execute(f"SELECT id FROM {table_name} WHERE name = ?", (flux_name,))
    flux_id = cur.fetchone()[0]

    update_sql = (f"UPDATE standard_variable SET standard_name = ?, standard_description = ?, standard_unit = ?, "
                  f"standard_dependencies = ?, standard_notes = ? WHERE standard_id = ? AND id = ?")

    for row in data['rows']:
        cur.execute(update_sql, (row[1], row[2], row[3], row[4], row[5], flux_id, row[0]))

    conn.commit()
    conn.close()
    print(f"Database updated from {in_json_file_path} successfully.")


def import_standards_from_list(in_session, in_function):
    save_standards_list = in_function()

    for save_standard_list in save_standards_list:
        save_standard = SaveStandards(name=save_standard_list['name'],
                                      description=save_standard_list['description'],
                                      function_name=save_standard_list['function_name'],
                                      folder_template=save_standard_list['folder_template'],
                                      file_templates=save_standard_list['file_templates'],
                                      extra_information=save_standard_list['extra_information'])
        standard_exists = in_session.query(SaveStandards).filter_by(name=save_standard.name,
                                                                    description=save_standard.description).first()
        if not standard_exists:
            in_session.add(save_standard)


def create_standard_variables_for_base_string(in_session, base_string, species_name, mag_field_models, in_epochs):
    description = construct_flux_description(base_string, species_name)
    standard_id = in_session.query(StandardsTable).first().id
    species_id = in_session.query(Species).filter_by(name=species_name).first().id
    related_flux = ''
    dependency_ids = []
    dependency_names = []

    add_standard_variable(in_session, [
        base_string,
        description,
        "Flux",
        "1/(cm2 s sr keV)",
        ""
    ],
    standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    base_string_err = f"{base_string}_err"
    description = construct_flux_description(base_string_err, species_name)

    add_standard_variable(in_session, [
        base_string_err,
        description,
        "Flux",
        "1/(cm2 s sr keV)",
        ""
    ],
                          standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    in_session.flush()

    add_standard_variable(in_session, [
        f"Energy_{base_string}",
        f"Energy channel labels for {base_string}.",
        "Energy",
        "keV",
        "Not always representative, check the energy channel bounds if something seems off."
    ],
                          standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    add_standard_variable(in_session, [
        f"Energy_{base_string}_minmax",
        f"Energy channel bounds for {base_string}.",
        "Energy",
        "keV",
        ""
    ],
                          standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    for index in range(1, 11):
        index_str = str(index).zfill(2)
        add_standard_variable(in_session, [
            f"FLAG_{index_str}_{base_string}",
            f"Flag {index} for {base_string}, see the FLAG_{index_str}_{base_string}_definition for details.",
            "FlagsModes",
            "",
            f"See FLAG_{index_str}_{base_string}_definition for details."
        ],
                              standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

        add_standard_variable(in_session, [
            f"FLAG_{index_str}_{base_string}_definition",
            f"Definitions for Flag {index} for {base_string}.",
            "FlagsModes",
            "",
            f"See FLAG_{index_str}_{base_string} for flag values."
        ],
                              standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    add_standard_variable(in_session, [
        f"PSD_{base_string}",
        f"Adiabatic phase space density for {base_string}.",
        "PhaseSpaceDensity",
        "s3 / (m6 kg3)",
        ""
    ],
                          standard_id, '', species_id, related_flux, dependency_ids, dependency_names)

    in_session.commit()
