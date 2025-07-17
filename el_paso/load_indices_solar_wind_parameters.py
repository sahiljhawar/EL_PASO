import os
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd
import scipy as sp
from astropy import units as u

import data_management.io as dm_io
import el_paso as ep
from el_paso.utils import enforce_utc_timezone


@overload
def load_indices_solar_wind_parameters(start_time:datetime,
                                       end_time:datetime,
                                       requested_outputs:Iterable[Literal["Kp", "SW_speed", "SW_density", "Dst", "Pdyn", "IMF_Bz", "IMF_By"]],
                                       target_time_variable:None=None,
                                       ) -> dict[str, tuple[ep.Variable, ep.Variable]]: ...

@overload
def load_indices_solar_wind_parameters(start_time:datetime,
                                       end_time:datetime,
                                       requested_outputs:Iterable[Literal["Kp", "SW_speed", "SW_density", "Dst", "Pdyn", "IMF_Bz", "IMF_By"]],
                                       target_time_variable:ep.Variable,
                                       ) -> dict[str, ep.Variable]: ...

def load_indices_solar_wind_parameters(start_time:datetime,
                                       end_time:datetime,
                                       requested_outputs:Iterable[Literal["Kp", "SW_speed", "SW_density", "Dst", "Pdyn", "IMF_Bz", "IMF_By"]],
                                       target_time_variable:ep.Variable|None=None,
                                       ) -> dict[str, tuple[ep.Variable, ep.Variable]] | dict[str, ep.Variable]:

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    if not isinstance(requested_outputs, list):
        msg = "requested_outputs must be a list of strings!"
        raise TypeError(msg)

    result_dict:dict[str, tuple[ep.Variable, ep.Variable]] | dict[str, ep.Variable] = {}

    base_data_path = Path(os.getenv("HOME")) / ".elpaso"

    for requested_output in requested_outputs:

        match requested_output:

            case "Kp":
                kp_model_order = [dm_io.kp.KpOMNI(base_data_path / "OMNI_low_res"), dm_io.kp.KpNiemegk(base_data_path/"KpNiemegk")]
                output_df = dm_io.kp.read_kp_from_multiple_models(start_time, end_time, model_order=kp_model_order, download=True)

                result = _create_variables_from_data_frame(output_df, "kp", target_time_variable, "previous")

            case "Dst":
                output_df = dm_io.dst.DSTOMNI(base_data_path / "OMNI_low_res").read(start_time, end_time, download=True)

                result = _create_variables_from_data_frame(output_df, "dst", target_time_variable, "linear")

            case "Pdyn":
                sw_model_order = [dm_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res")]
                output_df = dm_io.solar_wind.read_solar_wind_from_multiple_models(start_time-timedelta(hours=1), end_time+timedelta(hours=1), model_order=sw_model_order, download=True)
                output_df["pdyn"] = output_df["pdyn"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "pdyn", target_time_variable, "linear")

            case "IMF_Bz":
                sw_model_order = [dm_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res")]
                # we request two additional hours for interpolation 
                output_df = dm_io.solar_wind.read_solar_wind_from_multiple_models(start_time-timedelta(hours=1),
                                                                                  end_time+timedelta(hours=1),
                                                                                  model_order=sw_model_order,
                                                                                  download=True)
                output_df["bz_gsm"] = output_df["bz_gsm"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "bz_gsm", target_time_variable, "linear")

            case "IMF_By":
                # we request two additional hours for interpolation
                sw_model_order = [dm_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res")]
                output_df = dm_io.solar_wind.read_solar_wind_from_multiple_models(start_time-timedelta(hours=1),
                                                                                  end_time+timedelta(hours=1),
                                                                                  model_order=sw_model_order,
                                                                                  download=True)
                output_df["by_gsm"] = output_df["by_gsm"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "by_gsm", target_time_variable, "linear")

            case "SW_speed":
                # we request two additional hours for interpolation
                sw_model_order = [dm_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res")]
                output_df = dm_io.solar_wind.read_solar_wind_from_multiple_models(start_time-timedelta(hours=1),
                                                                                  end_time+timedelta(hours=1),
                                                                                  model_order=sw_model_order,
                                                                                  download=True)
                output_df["speed"] = output_df["speed"].interpolate(method="spline", order=3).ffill().bfill()

                result = _create_variables_from_data_frame(output_df, "speed", target_time_variable, "linear")

            case "SW_density":
                # we request two additional hours for interpolation
                sw_model_order = [dm_io.solar_wind.SWOMNI(base_data_path / "OMNI_high_res")]
                output_df = dm_io.solar_wind.read_solar_wind_from_multiple_models(start_time-timedelta(hours=1),
                                                                                  end_time+timedelta(hours=1),
                                                                                  model_order=sw_model_order,
                                                                                  download=True)
                output_df["proton_density"] = output_df["proton_density"].interpolate(method="spline", order=3)

                result = _create_variables_from_data_frame(output_df, "proton_density", target_time_variable, "linear")

            case _:
                msg = f"Requested invalid output: {requested_output}!"
                raise ValueError(msg)

        result_dict[requested_output] = result # type: ignore

    return result_dict

def _create_variables_from_data_frame(df_in:pd.DataFrame,
                                      data_key:str,
                                      target_time_variable:ep.Variable|None,
                                      time_interp_method:str) -> ep.Variable|tuple[ep.Variable,ep.Variable]:

    data_var = ep.Variable(data=df_in[data_key].values, original_unit=u.dimensionless_unscaled)
    timestamps = df_in.index.astype(np.int64)//10**9 # convert from ns to s
    time_var = ep.Variable(data=timestamps, original_unit=u.posixtime)

    if target_time_variable is None:
        result = (data_var, time_var)
    else:
        f = sp.interpolate.interp1d(time_var.get_data(), data_var.get_data(), kind=time_interp_method)
        data_var.set_data(f(target_time_variable.get_data(u.posixtime)), "same")
        result = data_var

    return result
