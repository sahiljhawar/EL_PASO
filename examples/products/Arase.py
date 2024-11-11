from datetime import datetime, timedelta, timezone

from astropy import units as u
import numpy as np
from IRBEM import Coords

from el_paso.classes import Product
from el_paso.classes import SourceFile
from el_paso.classes import Variable, TimeVariable, TimeBinMethod, VariableMetadata, DerivedVariable
from el_paso.save_standards.data_org import DataorgPMF
from el_paso.post_process.fold_pitch_angles_and_flux import fold_pitch_angles_and_flux

class AraseMepi(Product):

    def __init__(self, save_data_dir, download_data_dir, irbem_lib_path, magnetic_field_model='T89'):
        """Initializes the Arase HEP Realtime product."""
        super().__init__(name='Arase_Mepi', irbem_lib_path=irbem_lib_path)
        
        # Part 0: specify a save standard; this also fixes the variable names we have to use

        varnames = {}
        varnames['time'] = 'Epoch'
        varnames['Energy'] = 'Energy_FPDU'
        varnames['Flux'] = 'FPDU'
        varnames['PA_local'] = 'PA_local_FPDU'
        varnames['xGEO'] = 'xGEO'
        varnames['PSD'] = 'PSD_FPDU'
        varnames['PA_eq'] = 'PA_eq_T89'
        varnames['MLT'] = 'MLT_T89'
        varnames['Lstar'] = 'Lstar_T89_irbem'
        varnames['Lm'] = 'Lm_T89_irbem'
        varnames['PSD'] = 'PSD_FPDU'
        varnames['B_eq'] = 'B_eq_T89'
        varnames['R_eq'] = 'R_eq_T89'
        varnames['B_local'] = 'B_local_T89'
        varnames['InvMu'] = 'InvMu_T89'
        varnames['InvK'] = 'InvK_T89'
        self.default_save_standard = DataorgPMF(mission='Arase', source='arase', instrument='mepi',
                                                save_text_segments=[save_data_dir, 'arase',
                                                                    'n4', '4', 'T89', 'ver4'], varnames=varnames)

        # Part 1: specify source files to extract variables

        # Flux file
        time_var_flux = TimeVariable(name_or_column_in_file='epoch', original_unit=u.epoch_tt2000, standard_name='Epoch_posixtime')

        variables_to_extract_flux =  {
            'Epoch': time_var_flux, 
            'Energy_FPDU': Variable(name_or_column_in_file='FIDU_Energy', standard_name='Energy_FPDU',
                     original_unit=u.eV, target_unit=u.MeV, time_variable=None),
            'FPDU': Variable(name_or_column_in_file='FPDU', standard_name='FPDU',
                     original_unit='', time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var_flux),
            'PA_local_FPDU': Variable(name_or_column_in_file='FIDU_Alpha', original_unit=u.deg, target_unit='', standard_name='PA_local', time_variable=None),
        }

        data_path_stem = f"{download_data_dir}/arase/mepi/l3/pap/YYYY/"
        file_name_stem = "erg_mepi_l3_pa_YYYYMMDD*.cdf"

        source_file_flux = SourceFile(
            download_url="https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/mepi/l3/pa/YYYY/MM/",
            download_arguments_prefixes="-r -np -N -nH -e robots=off --cut-dirs=10 --accept 'erg_mepi_l3_pa_YYYYMMDD*.cdf'",
            download_arguments_suffixes=f"-P {data_path_stem}",
            download_path=f"{data_path_stem}{file_name_stem}",
            variables_to_extract=variables_to_extract_flux
        )

        # Position file
        time_var_position = TimeVariable(name_or_column_in_file='epoch', original_unit=u.epoch_tt2000, standard_name='Epoch_posixtime')

        variables_to_extract_position =  {
            'Epoch_pos': time_var_position, 
            'xGEO': Variable(name_or_column_in_file='pos_gsm', standard_name='xGEO',
                     original_unit=u.RE, target_unit=u.RE, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var_position)
        }

        data_path_stem = f"{download_data_dir}/arase/orb/def/YYYY/"
        file_name_stem = "erg_orb_l2_YYYYMMDD_v04.cdf"

        source_file_pos = SourceFile(
            download_url="https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/orb/def/YYYY/",
            download_arguments_prefixes="-r -np -N -nH -e robots=off --cut-dirs=10 --accept 'erg_orb_l2_YYYYMMDD_v04.cdf'",
            download_arguments_suffixes=f"-P {data_path_stem}",
            download_path=f"{data_path_stem}{file_name_stem}",
            variables_to_extract=variables_to_extract_position
        )

        self.source_files = [source_file_flux, source_file_pos]

        # Part 2: time binning
        self.perform_time_binning = True
        self.time_binning_cadence = timedelta(minutes=5)

        # Part 3: specify derived variables

        self.derived_variables = {
            'PSD_FPDU': DerivedVariable(standard_name='PSD_FPDU', target_unit='', time_variable=time_var_flux),
            'B_local_T89': DerivedVariable(standard_name=f'B_local_T89', target_unit=u.nT, time_variable=time_var_flux),
            'MLT_T89': DerivedVariable(standard_name=f'MLT_T89', target_unit=u.hour, time_variable=time_var_flux),
            'R_eq_T89': DerivedVariable(standard_name=f'R_eq_T89', target_unit=u.RE, time_variable=time_var_flux),
            'B_eq_T89': DerivedVariable(standard_name=f'B_eq_T89', target_unit=u.nT, time_variable=time_var_flux),
            'Lstar_T89': DerivedVariable(standard_name=f'Lstar_T89', target_unit='', time_variable=time_var_flux),
            'PA_eq_T89': DerivedVariable(standard_name=f'PA_eq_T89', target_unit='', time_variable=time_var_flux),
            'invMu_T89': DerivedVariable(standard_name=f'invMu_T89', target_unit=u.MeV*u.G, time_variable=time_var_flux),
            'invK_T89': DerivedVariable(standard_name=f'invK_T89', target_unit=u.RE*u.G**0.5, time_variable=time_var_flux)
        }

    def process_original_file(self) -> None:
        """Processes the variables from the original files for the product."""

        Flux_arr = self.extracted_variables['FPDU'].data_content
        Flux_arr[Flux_arr <= 0] = np.nan
        self.extracted_variables['FPDU'].data_content = Flux_arr

        # convert from GSM to GEO
        onera_lib_file = f"IRBEM/libirbem.so"
        model_coord = Coords(path=onera_lib_file)
        
        time_pos_arr = self.extracted_variables['Epoch_pos'].data_content
        xGSM_arr     = self.extracted_variables['xGEO'].data_content

        # convert time_array to datenums for transform function
        datetime_pos_arr = np.array([datetime.fromtimestamp(dt, tz=timezone.utc) for dt in time_pos_arr])
        xGEO_arr = model_coord.transform(datetime_pos_arr, xGSM_arr, 'GSM', 'GEO')
        self.extracted_variables['xGEO'].data_content = xGEO_arr

    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""

        # compute_mu_product(self, 'proton', 'Energy_FPDU', 'alpha_eq',
        #                    'B_eq', 'Epoch', 'PA_local_FPDU', fieldmodel)
        # compute_psd_product(self, 'proton', 'FPDU', 'Energy_FPDU', 'Epoch', 'PA_local_FPDU')


    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""
        fold_pitch_angles_and_flux(self)