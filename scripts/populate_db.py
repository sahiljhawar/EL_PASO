# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from el_paso.metadata.models import Base, StandardsTable, StandardVariableTypes, MagFieldModels, CoordinateSystems, Epochs
from el_paso.metadata.utils import add_species, add_standard_variable


def main():
    
    el_paso_dir = Path(os.environ.get('HOME')) / '.el_paso'
    el_paso_dir.mkdir(exist_ok=True)

    database_path = el_paso_dir / 'metadata_database.db'
    if database_path.exists():
        database_path.unlink()

    engine = create_engine(f'sqlite:///{database_path}')
    Base.metadata.create_all(engine)

    in_session = sessionmaker(bind=engine)
    session = in_session()

    standards = [
        StandardsTable(name='Internal', definition='First version of our internal data standard',
                       function_name='internal_standard.py')
    ]
    for standard in standards:
        exists = session.query(StandardsTable).filter_by(name=standard.name, definition=standard.definition).first()
        if not exists:
            session.add(standard)

    standard_variable_types = [
        StandardVariableTypes(name='Epoch', definition='variables related to time'),
        StandardVariableTypes(name='Flux',
                              definition='variables related to flux, such as flux, uncertainty of flux, energy flux, '
                                         'uncertainty of energy flux'),
        StandardVariableTypes(name='Position',
                              definition='variables related to position, such as coordinates or magnetic local time'),
        StandardVariableTypes(name='Energy',
                              definition='variables related to energy channel definitions, '
                                         'such as energy channels or upper/lower bounds of energy channels'),
        StandardVariableTypes(name='Bfield',
                              definition='variables related to B field, '
                                         'such as local/equatorial B field vector or field magnitude'),
        StandardVariableTypes(name='InvariantL', definition='variables related to L, such as L, Lm, Lstar, Lbasic'),
        StandardVariableTypes(name='InvariantMuKI', definition='variables related to mu, K, I invariants'),
        StandardVariableTypes(name='PitchAngle', definition='variables related to pitch angle'),
        StandardVariableTypes(name='FlagsModes',
                              definition='variables related to flags and operating mode descriptions'),
        StandardVariableTypes(name='PhaseSpaceDensity',
                              definition='variables related to adiabatic phase space density'),
        StandardVariableTypes(name='SolarWindParameters',
                              definition='variables related to solar wind parameters, including derived indices')
    ]
    for standard_variable_type in standard_variable_types:
        exists = session.query(StandardVariableTypes).filter_by(name=standard_variable_type.name,
                                                                definition=standard_variable_type.definition).first()
        if not exists:
            session.add(standard_variable_type)

    mag_fields = [
        MagFieldModels(name='observations', short_name='observations', long_name='Observations', irbem_internal=0,
                       irbem_external=0, default_options='0, 0, 9, 9, 0', standard_tag='OBS',
                       description='Observations'),
        MagFieldModels(name='dipole', short_name='dipole', long_name='Eccentric tilted dipole', irbem_internal=1,
                       irbem_external=0, default_options='0, 0, 9, 9, 1', standard_tag='DIP',
                       description='Eccentric tilted dipole'),
        MagFieldModels(name='IGRF', short_name='IGRF', long_name='IGRF', irbem_internal=0, irbem_external=0,
                       default_options='0, 0, 9, 9, 0', standard_tag='IGRF', description='IGRF'),
        MagFieldModels(name='OP77Q', short_name='OP77Q', long_name='Olson-Pfitzer quiet time', irbem_internal=0,
                       irbem_external=5, default_options='0, 0, 9, 9, 0', standard_tag='OP77Q',
                       description='Olson & Pfitzer quiet [1977] (default - Valid for rGEO≤15. RE)'),
        MagFieldModels(name='T89', short_name='T89', long_name='T89', irbem_internal=0, irbem_external=4,
                       default_options='0, 0, 9, 9, 0', standard_tag='T89',
                       description='Tsyganenko [1989c] (uses 0≤Kp≤9 - Valid for rGEO≤70. RE)'),
        MagFieldModels(name='T01s', short_name='T01s', long_name='T01s', irbem_internal=0, irbem_external=10,
                       default_options='0, 0, 9, 9, 0', standard_tag='T01S',
                       description='Tsyganenko [2001] storm (uses Dst, Pdyn, ByIMF, BzIMF, G2, G3 - '
                                   'there is no upper or lower limit for those inputs - Valid for xGSM≥-15. RE)'),
        MagFieldModels(name='T04s', short_name='T04s', long_name='T04s', irbem_internal=0, irbem_external=11,
                       default_options='0, 0, 9, 9, 0', standard_tag='T04S',
                       description='Tsyganenko [2004] storm (uses Dst, Pdyn, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6 - '
                                   'there is no upper or lower limit for those inputs - Valid for xGSM≥-15. RE)')
    ]
    for mag_field in mag_fields:
        exists = session.query(MagFieldModels).filter_by(name=mag_field.name, short_name=mag_field.short_name).first()
        if not exists:
            session.add(mag_field)
            for standard in standards:
                species_id = ''
                related_flux = ''
                dependency_ids = ''
                dependency_names = ''
                add_standard_variable(session, [
                    f"MLT_{mag_field.standard_tag}",
                    f"Magnetic local time from the {mag_field.short_name} magnetic field model.",
                    "MLT",
                    "hour",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)
                add_standard_variable(session, [
                    f"PA_local",
                    f"Local pitch angle(s).",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"PA_eq_{mag_field.standard_tag}",
                    f"Equatorial pitch angle(s) from {mag_field.short_name} magnetic field model.",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"PA_eq_real_{mag_field.standard_tag}",
                    f"Equatorial pitch angle(s) from B field measurements and {mag_field.short_name}.",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"PA_oa_local_{mag_field.standard_tag}",
                    f"Local pitch angle(s) for opening angles from {mag_field.short_name}.",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"PA_oa_eq_{mag_field.standard_tag}",
                    f"Equatorial pitch angle(s) for opening angles from {mag_field.short_name}.",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"PA_oa_eq_real_{mag_field.standard_tag}",
                    f"Equatorial pitch angle(s) for opening angles from B field measurements and "
                    f"{mag_field.short_name}.",
                    "PitchAngle",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"B_local_{mag_field.standard_tag}",
                    f"Magnitude of local B field from {mag_field.short_name}.",
                    "Bfield",
                    "nT",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"B_eq_{mag_field.standard_tag}",
                    f"Magnitude of B field on the same field line at the equator from {mag_field.short_name}.",
                    "Bfield",
                    "nT",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"B_fofl_{mag_field.standard_tag}",
                    f"Magnitude of local B field at the footpoint of the field line from {mag_field.short_name}.",
                    "Bfield",
                    "nT",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"L_{mag_field.standard_tag}_irbem",
                    f"local L from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantL",
                    "RE",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"L_{mag_field.standard_tag}_files",
                    f"local L from {mag_field.short_name}, from files from the data provider.",
                    "InvariantL",
                    "RE",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"Lm_{mag_field.standard_tag}",
                    f"McIlwain's L parameter from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantL",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"Lm_{mag_field.standard_tag}_files",
                    f"McIlwain's L parameter from {mag_field.short_name}, from files from the data provider.",
                    "InvariantL",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"Lm_{mag_field.standard_tag}_orlova",
                    f"McIlwain's L parameter from {mag_field.short_name}, computed with IRBEM and "
                    f"Ksenia Orlova's modification at GFZ.",
                    "InvariantL",
                    "deg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"Lstar_{mag_field.standard_tag}",
                    f"Roederer's Lstar parameter from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantL",
                    "",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"Lstar_{mag_field.standard_tag}_files",
                    f"Roederer's Lstar parameter from {mag_field.short_name}, from files from the data provider.",
                    "InvariantL",
                    "dg",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"R_eq_{mag_field.standard_tag}",
                    f"Radial distance of the point where the field line intersects the magnetic equator from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantL",
                    "RE",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"R_eq_{mag_field.standard_tag}_files",
                    f"Radial distance of the point where the field line intersects the magnetic equator from {mag_field.short_name}, from files from the data provider.",
                    "InvariantL",
                    "RE",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invMu_{mag_field.standard_tag}",
                    f"Invariant mu from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantMuKI",
                    "MeV / G",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invMu_files_{mag_field.standard_tag}",
                    f"Invariant mu from {mag_field.short_name}, from files from the data provider.",
                    "InvariantMuKI",
                    "MeV / G",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invMu_real_{mag_field.standard_tag}",
                    f"Invariant mu from {mag_field.short_name}, computed with IRBEM and "
                    f"real magnetic field measurements at GFZ.",
                    "InvariantMuKI",
                    "MeV / G",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invK_{mag_field.standard_tag}",
                    f"Invariant K from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantMuKI",
                    "RE * G^(1/2)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invK_files_{mag_field.standard_tag}",
                    f"Invariant K from {mag_field.short_name}, from files from the data provider.",
                    "InvariantMuKI",
                    "RE * G^(1/2)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"InvK_real_{mag_field.standard_tag}",
                    f"Invariant K from {mag_field.short_name}, computed with IRBEM and "
                    f"real magnetic field measurements at GFZ.",
                    "InvariantMuKI",
                    "RE * G^(1/2)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invI_{mag_field.standard_tag}_irbem",
                    f"Invariant I from {mag_field.short_name}, computed with IRBEM at GFZ.",
                    "InvariantMuKI",
                    "kg m^2/(s^2 nT)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invI_{mag_field.standard_tag}_files",
                    f"Invariant I from {mag_field.short_name}, from files from the data provider.",
                    "InvariantMuKI",
                    "kg m^2/(s^2 nT)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invI_real_{mag_field.standard_tag}_irbem",
                    f"Invariant I from {mag_field.short_name}, computed with IRBEM and "
                    f"real magnetic field measurements at GFZ.",
                    "InvariantMuKI",
                    "kg m^2/(s^2 nT)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"invI_real_{mag_field.standard_tag}_files",
                    f"Invariant I from {mag_field.short_name}, from files from the data provider "
                    f"based on real magnetic field measurements.",
                    "InvariantMuKI",
                    "kg m^2/(s^2 nT)",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)

                add_standard_variable(session, [
                    f"xGEO_{mag_field.standard_tag}",
                    f"",
                    "Position",
                    "RE",
                    ""
                ],
                                      standard.id, mag_field.id, species_id, related_flux, dependency_ids,
                                      dependency_names)


    coordinate_systems = [
        CoordinateSystems(name='GDZ', description='Position in geodetic coordinates',
                          definition='(geodetic; altitude, latitude, East longitude), '
                                     'Defined using a reference ellipsoid. '
                                     'Geodetic longitude is identical to GEO longitude. Both the altitude and latitude '
                                     'depend on the ellipsoid used. IRBEM uses the WGS84 reference ellipsoid. ',
                          abbreviation='GDZ', standard_unit='RE, deg, deg'),
        CoordinateSystems(name='GEO', description='Position in geocentric geographic coordinates',
                          definition='(geocentric geographic; Cartesian) , '
                                     'Earth-Centered and Earth-Fixed. X lies in the '
                                     'Earth''s equatorial plane (zero latitude) and intersects the Prime Meridian '
                                     '(zero longitude; Greenwich, UK). Z points to True North '
                                     '(roughly aligned with the instantaneous rotation axis). ',
                          abbreviation='GEO', standard_unit='RE'),
        CoordinateSystems(name='GSM', description='Position in geocentric solar magnetospheric coordinates',
                          definition='(geocentric solar magnetospheric; Cartesian), '
                                     'X points sunward from Earth''s center. '
                                     'The X-Z plane is defined to contain '
                                     'Earth''s dipole axis (positive North). ',
                          abbreviation='GSM', standard_unit='RE'),
        CoordinateSystems(name='GSE', description='Position in geocentric solar ecliptic coordinates',
                          definition='(geocentric solar ecliptic; Cartesian), X points sunward from Earth''s center. '
                                     'Y lies in the ecliptic plane of date, pointing in the anti-orbit direction. '
                                     'Z is parallel to the ecliptic pole of date.  ',
                          abbreviation='GSE', standard_unit='RE'),
        CoordinateSystems(name='SM', description='Position in solar magnetic coordinates',
                          definition='(solar magnetic; Cartesian), Z is aligned with the centered dipole axis of date '
                                     '(positive North), '
                                     'and Y is perpendicular to both the Sun-Earth line and the dipole '
                                     'axis. X is therefore is not aligned with the Sun-Earth line and '
                                     'SM is a rotation about Y from GSM.',
                          abbreviation='SM', standard_unit='RE'),
        CoordinateSystems(name='GEI', description='Position in geocentric equatorial inertial coordinates',
                          definition='(geocentric equatorial inertial, of Date; '
                                     'Cartesian), X points from Earth toward the '
                                     'equinox of date (first point of Aries; '
                                     'position of the Sun at the vernal equinox). '
                                     'Z is parallel to the instantaneous rotation axis of the Earth. ',
                          abbreviation='GEI', standard_unit='RE'),
        CoordinateSystems(name='MAG', description='Position in geomagnetic coordinates',
                          definition='(geomagnetic; Cartesian), Z is parallel to Earth''s centered dipole axis '
                                     '(positive North). Y is the intersection between Earth''s equator and the '
                                     'geographic meridian 90 deg east of the meridian containing the dipole axis.',
                          abbreviation='MAG', standard_unit='RE'),
        CoordinateSystems(name='SPH', description='Position in spherical geocentric geographic coordinates',
                          definition='(GEO in spherical; radial distance, latitude, East longitude), '
                                     'Geocentric geographic coordinates (GEO system) expressed in spherical '
                                     'instead of Cartesian.',
                          abbreviation='SPH', standard_unit='RE, deg, deg'),
        CoordinateSystems(name='RLL', description='Position in spherical geodetic coordinates',
                          definition='(geodetic; radial distance, geodetic latitude, East longitude), '
                                     'A re-expression of geodetic (GDZ) coordinates using radial distance instead of '
                                     'altitude above the reference ellipsoid. Note that the latitude is still geodetic '
                                     'latitude and is therefore not interchangeable with SPH. ',
                          abbreviation='RLL', standard_unit='RE, deg, deg'),
        CoordinateSystems(name='HEE', description='Position in heliocentric Earth ecliptic coordinates',
                          definition='(heliocentric Earth ecliptic; Cartesian), '
                                     'Origin is solar center; '
                                     'X points towards the Earth, and Z is perpendicular to the '
                                     'plane of Earth''s orbit (positive North). This system is fixed with respect '
                                     'to the Earth-Sun line. ',
                          abbreviation='HEE', standard_unit='RE'),
        CoordinateSystems(name='HAE', description='Position in heliocentric Aries ecliptic coordinates',
                          definition='(heliocentric Aries ecliptic; Cartesian), '
                                     'Origin is solar center. Z is perpendicular to the plane of Earth''s orbit '
                                     '(positive North) and '
                                     'X points towards the equinox of date (first point of Aries). ',
                          abbreviation='HAE', standard_unit='RE'),
        CoordinateSystems(name='HEEQ', description='Position in heliocentric Earth equatorial coordinates',
                          definition='(heliocentric Earth equatorial; Cartesian), '
                                     'Origin is solar center. '
                                     'Z is parallel to the Sun''s rotation axis (positive North) '
                                     'and X points towards the intersection of the solar equator and '
                                     'solar central meridian as seen from Earth. ',
                          abbreviation='HEEQ', standard_unit='RE'),

    ]
    for coordinate_system in coordinate_systems:
        exists = session.query(CoordinateSystems).filter_by(name=coordinate_system.name,
                                                            abbreviation=coordinate_system.abbreviation).first()
        if not exists:
            session.add(coordinate_system)
            for standard in standards:
                add_standard_variable(session, [
                    f"x{coordinate_system.abbreviation}",
                    f"{coordinate_system.description}",
                    "Position",
                    f"{coordinate_system.standard_unit}",
                    ""
                ],
                                      standard.id, '', '', '', '',
                                      '')

    epochs = [
        Epochs(name='CDF tt2000', epoch_name='tt2000', description='CDF tt2000'),
        Epochs(name='CDF 8-bit', epoch_name='cdf8', description='CDF epoch, 8-bit precision'),
        Epochs(name='CDF 16-bit', epoch_name='cdf16', description='CDF epoch, 16-bit precision'),
        Epochs(name='Matlab datenum', epoch_name='datenum', description='Matlab datenum'),
        Epochs(name='POSIX time', epoch_name='posixtime', description='POSIX time'),
        Epochs(name='Julian date', epoch_name='juliandate', description='Julian date'),
        Epochs(name='Second of day', epoch_name='secondofday', description='Second of day'),
    ]
    for epoch in epochs:
        exists = session.query(Epochs).filter_by(name=epoch.name, epoch_name=epoch.epoch_name).first()
        if not exists:
            session.add(epoch)
            for standard in standards:
                add_standard_variable(session, [
                    f"Epoch_{epoch.epoch_name}",
                    f"{epoch.name}",
                    "Epoch",
                    f"{epoch.epoch_name}",
                    ""
                ],
                                      standard.id, '', '', '', '',
                                      '')

    # Add solar wind parameters
    for standard in standards:
        add_standard_variable(session, [
                    f"Bartels_rotation_number",
                    f"Bartels rotation number",
                    "SolarWindParameters",
                    "",
                    ""
                ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Field_magnitude_average",
            f"Field Magnitude Average |B|       1/N SUM |B|",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Average_field_vector",
            f"Magnitude of Average Field Vector sqrt(Bx^2+By^2+Bz^2)",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Average_field_vector_lat_angle",
            f"Lat.Angle of Aver. Field Vector, deg (GSE coords)",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Average_field_vector_long_angle",
            f"Long.Angle of Aver. Field Vector, deg (GSE coords)",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Bx_GSE",
            f"Bx GSE, GSM for solar wind",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"By_GSE",
            f"By GSE for solar wind",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Bz_GSE",
            f"Bz GSE for solar wind",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"By_GSM",
            f"By GSM for solar wind",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Bz_GSM",
            f"Bz GSM for solar wind",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_B_mag",
            f"sigma|B|, RMS Standard Deviation in average magnitude (word 10)",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_B",
            f"sigma B, RMS Standard Deviation in field vector",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_Bx",
            f"sigma Bx, RMS Standard Deviation in GSE X-component average",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_By",
            f"sigma By, RMS Standard Deviation in GSE Y-component average",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_Bz",
            f"sigma Bz, RMS Standard Deviation in GSE Z-component average",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"T_proton",
            f"Proton temperature",
            "SolarWindParameters",
            "K",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Dens_proton",
            f"Proton density",
            "SolarWindParameters",
            "1/cm3",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_speed",
            f"Plasma (Flow) speed, km/s",
            "SolarWindParameters",
            "km/s",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_velocity_x",
            f"Vx Velocity, km/s, GSE",
            "SolarWindParameters",
            "km/s",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_velocity_y",
            f"Vy Velocity, km/s, GSE",
            "SolarWindParameters",
            "km/s",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_velocity_z",
            f"Vz Velocity, km/s, GSE",
            "SolarWindParameters",
            "km/s",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_long_angle",
            f"Plasma Flow Long. Angle, deg, quasi-GSE*",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_lat_angle",
            f"Plasma  Flow Lat. Angle, deg, GSE*",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"na_over_np",
            f"Na/Np, Alpha/Proton ratio",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Flow_pressure",
            f"Flow Pressure, P (nPa) = (1.67/10**6) * Np*V**2 * (1+ 4*Na/Np)",
            "SolarWindParameters",
            "nPa",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_T",
            f"Standard deviation for proton temperature",
            "SolarWindParameters",
            "K",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_N",
            f"Standard deviation for proton density",
            "SolarWindParameters",
            "1/cm3",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_V",
            f"Standard deviation for flow speed",
            "SolarWindParameters",
            "km/s",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_phi",
            f"Standard deviation for flow longitudinal angle",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_theta",
            f"Standard deviation for flow latitudinal angle",
            "SolarWindParameters",
            "deg",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sigma_na_over_np",
            f"Standard deviation for alpha/proton ratio",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"E_field",
            f"Electric field, -[V(km/s) * Bz (nT; GSM)] * 10**-3.",
            "SolarWindParameters",
            "mV/m",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Plasma_beta",
            f"Plasma beta, Beta = [(T*4.16/10**5) + 5.34] * Np / B**2",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Alfven_mach_number",
            f"Alfven mach number, Ma = (V * Np**0.5) / 20 * B",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Kp",
            f"Kp, Planetary Geomagnetic Activity Index",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"sunspot_R",
            f"R, Sunspot number (new version 2)",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Dst",
            f"DST (dynamic storm time) Index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"Kp",
            f"Kp (kennzifer planetarisch/potsdam) index",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"AE",
            f"AE auroral electrojet Index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"AP",
            f"AP index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"f10_7",
            f"f10.7_index, ( sfu = 10-22W.m-2.Hz-1)",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"PC_N",
            f"PC(N) index",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"AL",
            f"AL-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"AU",
            f"AU-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"MS_mach_number",
            f"Magnetosonic mach number= = V/Magnetosonic_speed",
            "SolarWindParameters",
            "",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"SYMD",
            f"SYM/D-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"SYMH",
            f"SYM/H-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"ASYD",
            f"ASY/D-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')
        add_standard_variable(session, [
            f"ASYH",
            f"ASY/H-index",
            "SolarWindParameters",
            "nT",
            ""
        ],
                              standard.id, '', '', '', '',
                              '')

    # Add particle species
    add_species(session, 'electron', 'E', 0.511, 'Electrons')
    add_species(session, 'proton', 'P', 938, 'Protons')
    add_species(session, 'helium', 'HE', 3727.6, 'All helium ion species combined')
    add_species(session, 'helium+', 'HE1', 3727.6, 'Helium+ ions')
    add_species(session, 'helium++', 'HE2', 3727.6, 'Helium++ ions')
    add_species(session, 'oxygen', 'O', 14899.16083801, 'All oxygen ion species combined')

    session.commit()
    session.close()


if __name__ == "__main__":
    main()
