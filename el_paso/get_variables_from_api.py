from datetime import datetime
from typing import Literal

import cdasws

import el_paso as ep


def get_variables_from_api(start_time:datetime,
                           end_time:datetime,
                           dataset_name:str,
                           api_type:Literal["cdasWs"]="cdasWs") -> dict[str,ep.Variable]:

    match api_type:
        case "cdasWs":
            data = _cdasws_api_access(start_time, end_time, dataset_name)

def _cdasws_api_access(start_time:datetime,
                       end_time:datetime,
                       dataset_name:str):

    cdas = cdasws.CdasWs()

    time_interval = cdasws.TimeInterval(start_time, end_time)

    status, data = cdas.get_data(dataset_name,
                                 ["ALL-VARIABLES"],
                                 time_interval,
                                 data_representation=cdasws.DataRepresentation.XARRAY)

    if status.apiStatusCode == 200:
        return data
    else:
        msg = f"Error retrieving data: {status.apiStatusPhrase}"
        raise ValueError(msg)
