from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from astropy import units as u

import el_paso as ep


def process_ect_combined(start_time:datetime,
                         end_time:datetime,
                         sat_str:Literal["a", "b"],
                         irbem_lib_path:str|Path,
                         mag_field:Literal["T89", "T96", "TS04"],
                         raw_data_path:str|Path = ".",
                         processed_data_path:str|Path = ".",
                         num_cores:int=4):

    irbem_lib_path = Path(irbem_lib_path)
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)

    file_name_stem = "rbsp" + sat_str + "_ect-elec-L3_YYYYMMDD_.{6}.cdf"

    ep.download(start_time, end_time,
                save_path=raw_data_path,
                download_url=f"https://rbsp-ect.newmexicoconsortium.org/data_pub/rbsp{sat_str}/ECT/level3/YYYY/",
                file_name_stem=file_name_stem,
                file_cadence="daily",
                method="request",
                skip_existing=True)

    extraction_infos = [
        ep.ExtractionInfo(
            result_key="Epoch",
            name_or_column="Epoch",
            unit=ep.units.cdf_epoch,
        ),
        ep.ExtractionInfo(
            result_key="Energy",
            name_or_column="FEDU_Energy",
            unit=u.keV,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="Pitch_angle",
            name_or_column="FEDU_Alpha",
            unit=u.deg,
            is_time_dependent=False,
        ),
        ep.ExtractionInfo(
            result_key="FEDU",
            name_or_column="FEDU",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="FEDU_quality",
            name_or_column="FEDU_Quality",
            unit=u.dimensionless_unscaled,
        ),
        ep.ExtractionInfo(
            result_key="FEDO",
            name_or_column="FEDO",
            unit=(u.cm**2 * u.s * u.sr * u.keV) ** (-1),
        ),
        ep.ExtractionInfo(
            result_key="xGEO",
            name_or_column="Position",
            unit=u.km,
        ),
    ]

    variables = ep.extract_variables_from_files(start_time, end_time, "daily",
                                                data_path=raw_data_path, file_name_stem=file_name_stem,
                                                extraction_infos=extraction_infos)

    time_bin_methods = {
        "xGEO": ep.TimeBinMethod.NanMean,
        "Energy": ep.TimeBinMethod.Repeat,
        "FEDU": ep.TimeBinMethod.NanMedian,
        "FEDU_Quality": ep.TimeBinMethod.NanMax,
        "FEDO": ep.TimeBinMethod.NanMedian,
        "Pitch_angle": ep.TimeBinMethod.Repeat,
    }

    binned_time_variable = ep.processing.bin_by_time(variables["Epoch"], variables=variables,
                                                    time_bin_method_dict=time_bin_methods,
                                                    time_binning_cadence=timedelta(minutes=5),
                                                    start_time=start_time, end_time=end_time)

    variables["FEDU"].transpose_data([0,2,1]) # making it having dimensions (time, energy, pitch angle)
    ep.processing.fold_pitch_angles_and_flux(variables["FEDU"],
                                            variables["Pitch_angle"])

    # not needed anymore
    del variables["Epoch"]

    indices_solar_wind = ep.load_indices_solar_wind_parameters(start_time,
                                                               end_time,
                                                               requested_outputs=["Kp", "Dst", "Pdyn", "IMF_Bz", "IMF_By", "SW_speed", "SW_density"],
                                                               target_time_variable=binned_time_variable)

    Kp = indices_solar_wind["Kp"].get_data()
    indices_solar_wind["Kp"].set_data(Kp, "same")

    # Calculate magnetic field variables
    irbem_options = [1, 1, 4, 4, 0]

    var_names_to_compute = ["B_local_" + mag_field,
                            "MLT_" + mag_field,
                            "B_eq_" + mag_field,
                            "R_eq_" + mag_field,
                            "PA_eq_" + mag_field]

    magnetic_field_variables = ep.processing.compute_magnetic_field_variables(time_var = binned_time_variable,
                                                                              xgeo_var = variables["xGEO"],
                                                                              var_names_to_compute = var_names_to_compute,
                                                                              irbem_lib_path = str(irbem_lib_path),
                                                                              irbem_options = irbem_options,
                                                                              num_cores = num_cores,
                                                                              pa_local_var = variables["Pitch_angle"],
                                                                              indices_solar_wind = indices_solar_wind)

    # saving_strategy = ep.saving_strategies.DataOrgStrategy(processed_data_path, "RBSP", "rbsp" + sat_str, "ect-combined", mag_field, ".mat")

    # variables_to_save = {
    #     "time": binned_time_variable,
    #     "Flux": variables["FEDU"],
    #     "xGEO": variables["xGEO"],
    #     "energy_channels": variables["Energy"],
    #     "alpha_local": variables["Pitch_angle"],
    #     "alpha_eq_model": magnetic_field_variables["PA_eq_" + mag_field],
    #     "Lstar": magnetic_field_variables["Lstar_" + mag_field],
    #     "MLT": magnetic_field_variables["MLT_" + mag_field],
    # }

    # ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)

    saving_strategy = ep.saving_strategies.MonthlyH5Strategy(processed_data_path, file_name_stem="ECT-combined", mag_field=mag_field)

    variables_to_save = {
        "time": binned_time_variable,
        "flux/FEDU": variables["FEDU"],
        "flux/FEDO": variables["FEDO"],
        "flux/energy": variables["Energy"],
        "flux/alpha_local": variables["Pitch_angle"],
        "flux/alpha_eq": magnetic_field_variables["PA_eq_T89"],
        f"position/{mag_field}/R0": magnetic_field_variables["R_eq_T89"],
        f"position/{mag_field}/MLT": magnetic_field_variables["MLT_T89"],
    }

    ep.save(variables_to_save, saving_strategy, start_time, end_time, binned_time_variable)

if __name__ == "__main__":

    start_time = datetime(2017, 4, 20, tzinfo=timezone.utc)
    end_time = datetime(2017, 4, 24, tzinfo=timezone.utc)

    process_ect_combined(start_time, end_time, "a", "../../IRBEM/libirbem.so", "T89",
                         raw_data_path=".", processed_data_path=".", num_cores=16)