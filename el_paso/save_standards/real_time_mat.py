from __future__ import annotations

from datetime import datetime
from pathlib import Path

from astropy import units as u

from el_paso.classes import Variable
from el_paso.classes.save_standard import OutputFile, SaveCadence, SaveStandard


class RealtimeMat(SaveStandard):
    def __init__(
        self,
        product_variable_names,
        mission: str = None,
        source: str = None,
        instrument: str = None,
        model: str = None,
        mfm: str = None,
        version: str = None,
        save_text_segments: list[str] = None,
        default_db: str = None,
        default_format: str = None,
    ):
        super().__init__(
            mission,
            source,
            instrument,
            model,
            mfm,
            version,
            save_text_segments,
            default_db,
            default_format,
            product_variable_names,
        )

        variable_names_in_file = [
            "newtime",
            "Energy",
            "Pitch_Angles",
            "alpha_loc",
            "alpha_eq",
            "Lstar",
            "Flux",
            "PSD",
            "Position_GEO",
            "InvMu",
            "InvK",
            "B_eq",
            "B_loc",
            "MLT",
        ]

        assert all(
            [x in list(product_variable_names.keys()) for x in variable_names_in_file]
        ), f"Variable name(s) {[x for x in variable_names_in_file if x not in list(product_variable_names.keys())]} not provided! Add them to the 'varnames' argument."


        self.save_cadence = SaveCadence.ONE_FILE

        self.output_files = [
            OutputFile(
                "realtime_flux",
                ["newtime", "Energy", "Pitch_Angles", "alpha_loc", "alpha_eq", "Lstar", "InvMu", "InvK", "Flux", "PSD", "B_eq", "B_loc", "MLT", "Position_GEO"],  # noqa: E501
                [u.datenum, u.MeV, u.deg, u.deg, u.deg, "", u.MeV/u.G, u.RE*u.G**(1/2), (u.cm**2*u.s*u.sr*u.keV)**(-1), u.s**3/(u.kg**3 * u.m**6), u.G, u.G, u.hour, u.km]),  # noqa: E501
        ]

    def get_saved_file_name(self, start_time: datetime, end_time: datetime, output_file: OutputFile, external_text:str|None=None) -> str:

        year_month = start_time.strftime("%Y%m")
        year_month_day = start_time.strftime("%Y%m%d")

        file_folder_name = Path(f"{self.save_text_segments[0]}/{year_month}")
        file_folder_name.mkdir(exist_ok=True)
        file_name = (
            f"{self.save_text_segments[0]}/{year_month}/{self.source}_{self.save_text_segments[1]}_"
            f"{year_month_day}_{self.save_text_segments[2]}_{self.save_text_segments[3]}.mat"
        )

        return file_name
