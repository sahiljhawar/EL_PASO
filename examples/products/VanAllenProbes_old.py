from datetime import timedelta

from astropy import units as u
import numpy as np

from el_paso.classes import Product
from el_paso.classes import SourceFile
from el_paso.classes import Variable, TimeVariable, TimeBinMethod, VariableMetadata, DerivedVariable
from el_paso.save_standards.data_org import DataorgPMF
from el_paso.save_standards.basic_standard import BasicStandard
from el_paso.post_process.fold_pitch_angles_and_flux import fold_pitch_angles_and_flux

class HOPE_electron(Product):

    def __init__(self, satellite_str, save_data_dir, download_data_dir, irbem_lib_path, num_cores=32):
        """Initializes the Arase HEP Realtime product."""
        super().__init__(irbem_lib_path=irbem_lib_path, num_cores=num_cores)
        
        assert satellite_str in ['rbspa', 'rbspb']

        # Part 0: specify a save standard; this also fixes the variable names we have to use

        varnames = {}
        varnames['time'] = 'Epoch'
        varnames['Energy'] = 'Energy_FEDU'
        varnames['Flux'] = 'FEDU'
        varnames['alpha_local'] = 'PA_local_FEDU'
        varnames['xGEO'] = 'xGEO'
        varnames['PSD'] = 'PSD_FEDU'
        varnames['alpha_eq_model'] = 'PA_eq_T89'
        varnames['MLT'] = 'MLT_T89'
        varnames['Lstar'] = 'Lstar_T89'
        varnames['Lm'] = 'Lm_T89'
        varnames['R_eq'] = 'R_eq_T89'
        varnames['PSD'] = 'PSD_FEDU'
        varnames['B_eq'] = 'B_eq_T89'
        varnames['B_local'] = 'B_local_T89'
        varnames['InvMu'] = 'invMu_T89'
        varnames['InvK'] = 'invK_T89'
        self.save_standard = DataorgPMF(mission='RBSP', source=satellite_str, instrument='HOPE-electron-l3', save_text_segments=[save_data_dir, satellite_str,
                                                                    'n4', '4', 'T89', 'ver4'], product_variable_names=varnames)

        #self.default_save_standard = BasicStandard()

        # Part 1: specify source files to extract variables

        time_var = TimeVariable(name_or_column_in_file='Epoch_Ele', original_unit=u.tt2000, standard_name='Epoch_posixtime')

        variables_to_extract =  {
            'Epoch': time_var, 
            'Energy_FEDU': Variable(name_or_column_in_file='HOPE_ENERGY_Ele', standard_name='Energy_FEDU', original_unit=u.eV, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var),
            'FEDU': Variable(name_or_column_in_file='FEDU', standard_name='FEDU',
                     original_unit=(u.cm**2*u.s*u.sr*u.keV)**(-1), time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var),
            'PA_local_FEDU': Variable(name_or_column_in_file='PITCH_ANGLE', original_unit=u.deg, standard_name='PA_local', time_variable=None),
            'xGEO': Variable(name_or_column_in_file='Position_Ele', standard_name='xGEO',
                     original_unit=u.km, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var)
        }

        data_path_stem = f"{download_data_dir}rbsp/{satellite_str}/hope/l3/pitchangle/YYYY/"
        file_name_stem = f'rbsp{satellite_str[-1]}_rel04_ect-hope-pa-l3_YYYYMMDD*.cdf'

        source_file = SourceFile(
            download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/{satellite_str}/l3/ect/hope/pitchangle/rel04/YYYY/",
            download_arguments_prefixes=f"-r -np -N -nH -e robots=off --cut-dirs=10 --accept '{file_name_stem}'",
            download_arguments_suffixes=f"-P {data_path_stem}",
            download_path=f"{data_path_stem}{file_name_stem}",
            variables_to_extract=variables_to_extract
        )

        self.source_files = [source_file]

        # Part 2: time binning
        self.perform_time_binning = True
        self.time_binning_cadence = timedelta(minutes=5)

        # Part 3: specify derived variables
        self.derived_variables = {
            'PSD_FEDU': DerivedVariable(standard_name='PSD_FEDU', time_variable=time_var),
            'B_local_T89': DerivedVariable(standard_name=f'B_local_T89', time_variable=time_var),
            'MLT_T89': DerivedVariable(standard_name=f'MLT_T89', time_variable=time_var),
            'R_eq_T89': DerivedVariable(standard_name=f'R_eq_T89', time_variable=time_var),
            'B_eq_T89': DerivedVariable(standard_name=f'B_eq_T89', time_variable=time_var),
            #'Lstar_T89': DerivedVariable(standard_name=f'Lstar_T89', time_variable=time_var),
            'PA_eq_T89': DerivedVariable(standard_name=f'PA_eq_T89', time_variable=time_var),
            'invMu_T89': DerivedVariable(standard_name=f'invMu_T89', time_variable=time_var),
            'invK_T89': DerivedVariable(standard_name=f'invK_T89', time_variable=time_var)
        }

    def process_original_file(self) -> None:
        """Processes the variables from the original files for the product."""

        Flux_arr = self.extracted_variables['FEDU'].data_content
        Flux_arr[Flux_arr <= 0] = np.nan
        Flux_arr = np.transpose(Flux_arr, (0,2,1))
        self.extracted_variables['FEDU'].data_content = Flux_arr

    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""

        # compute_mu_product(self, 'proton', 'Energy_FPDU', 'alpha_eq',
        #                    'B_eq', 'Epoch', 'PA_local_FPDU', fieldmodel)
        # compute_psd_product(self, 'proton', 'FPDU', 'Energy_FPDU', 'Epoch', 'PA_local_FPDU')

    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""

        fold_pitch_angles_and_flux(self)

class HOPE_proton(Product):

    def __init__(self, satellite_str, save_data_dir, download_data_dir, irbem_lib_path, num_cores=64):
        """Initializes the Arase HEP Realtime product."""
        super().__init__(irbem_lib_path=irbem_lib_path, num_cores=num_cores, save_cadence='monthly')
        
        assert satellite_str in ['rbspa', 'rbspb']

        # Part 0: specify a save standard; this also fixes the variable names we have to use

        varnames = {}
        varnames['time'] = 'Epoch'
        varnames['Energy'] = 'Energy_FPDU'
        varnames['Flux'] = 'FPDU'
        varnames['alpha_local'] = 'PA_local_FPDU'
        varnames['xGEO'] = 'xGEO'
        varnames['PSD'] = 'PSD_FPDU'
        varnames['alpha_eq_model'] = 'PA_eq_T89'
        varnames['MLT'] = 'MLT_T89'
        varnames['Lstar'] = 'Lstar_T89'
        varnames['Lm'] = 'Lm_T89'
        varnames['R_eq'] = 'R_eq_T89'
        varnames['PSD'] = 'PSD_FPDU'
        varnames['B_eq'] = 'B_eq_T89'
        varnames['B_local'] = 'B_local_T89'
        varnames['InvMu'] = 'invMu_T89'
        varnames['InvK'] = 'invK_T89'

        self.save_standard = DataorgPMF(mission='RBSP', source=satellite_str, instrument='HOPE-proton-l3', save_text_segments=[save_data_dir, satellite_str,
                                                                    'n4', '4', 'T89', 'ver4'], product_variable_names=varnames)

        #self.save_standard = BasicStandard(save_text_segments=['test'])

        # Part 1: specify source files to extract variables

        time_var = TimeVariable(name_or_column_in_file='Epoch_Ion', original_unit=u.tt2000, standard_name='Epoch_posixtime')

        variables_to_extract =  {
            'Epoch': time_var,
            'FPDU': Variable(name_or_column_in_file='FPDU', standard_name='FPDU',
                original_unit=(u.cm**2*u.s*u.sr*u.keV)**(-1), time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var),
            'Energy_FPDU': Variable(name_or_column_in_file='HOPE_ENERGY_Ion', standard_name='Energy_FPDU', original_unit=u.eV, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var),
            'PA_local_FPDU': Variable(name_or_column_in_file='PITCH_ANGLE', original_unit=u.deg, standard_name='PA_local', time_variable=None),
            'xGEO': Variable(name_or_column_in_file='Position_Ion', standard_name='xGEO',
                     original_unit=u.km, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var)
        }

        data_path_stem = f"{download_data_dir}rbsp/{satellite_str}/hope/l3/pitchangle/YYYY/"
        file_name_stem = f'rbsp{satellite_str[-1]}_rel04_ect-hope-pa-l3_YYYYMMDD*.cdf'

        source_file = SourceFile(
            download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/{satellite_str}/l3/ect/hope/pitchangle/rel04/YYYY/",
            download_arguments_prefixes=f"-r -np -N -nH -e robots=off --cut-dirs=10 --accept '{file_name_stem}'",
            download_arguments_suffixes=f"-P {data_path_stem}",
            download_path=f"{data_path_stem}{file_name_stem}",
            variables_to_extract=variables_to_extract
        )

        self.source_files = [source_file]

        # Part 2: time binning
        self.perform_time_binning = True
        self.time_binning_cadence = timedelta(minutes=5)

        # Part 3: specify derived variables
        self.derived_variables = {
            'PSD_FPDU': DerivedVariable(standard_name='PSD_FPDU', time_variable=time_var),
            'B_local_T89': DerivedVariable(standard_name=f'B_local_T89', time_variable=time_var),
            'MLT_T89': DerivedVariable(standard_name=f'MLT_T89', time_variable=time_var),
            'R_eq_T89': DerivedVariable(standard_name=f'R_eq_T89', time_variable=time_var),
            'B_eq_T89': DerivedVariable(standard_name=f'B_eq_T89', time_variable=time_var),
            'Lstar_T89': DerivedVariable(standard_name=f'Lstar_T89', time_variable=time_var),
            'PA_eq_T89': DerivedVariable(standard_name=f'PA_eq_T89', time_variable=time_var),
            'invMu_T89': DerivedVariable(standard_name=f'invMu_T89', time_variable=time_var),
            'invK_T89': DerivedVariable(standard_name=f'invK_T89', time_variable=time_var)
        }

    def process_original_file(self) -> None:
        """Processes the variables from the original files for the product."""

        print('Process original files')

        Flux_arr = self.extracted_variables['FPDU'].data_content
        Flux_arr[Flux_arr <= 0] = np.nan
        Flux_arr = np.transpose(Flux_arr, (0,2,1))
        self.extracted_variables['FPDU'].data_content = Flux_arr

    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""

        self.extracted_variables['Epoch'].convert_to_unit(u.datenum)
        self.derived_variables['B_local_T89'].convert_to_unit(u.G)
        self.derived_variables['B_eq_T89'].convert_to_unit(u.G)

    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""
        fold_pitch_angles_and_flux(self)

class RBSPICE_tofxeh(Product):

    def __init__(self, satellite_str, save_data_dir, download_data_dir, irbem_lib_path):
        """Initializes the Arase HEP Realtime product."""
        super().__init__(irbem_lib_path=irbem_lib_path)
        
        assert satellite_str in ['rbspa', 'rbspb']

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
        varnames['Lstar'] = 'Lstar_T89'
        varnames['Lm'] = 'Lm_T89'
        varnames['R_eq'] = 'R_eq_T89'
        varnames['PSD'] = 'PSD_FPDU'
        varnames['B_eq'] = 'B_eq_T89'
        varnames['B_local'] = 'B_local_T89'
        varnames['InvMu'] = 'invMu_T89'
        varnames['InvK'] = 'invK_T89'
        self.default_save_standard = DataorgPMF(mission='RBSP', source=satellite_str, instrument='RBSPICE-tofxeh-l3', save_text_segments=[save_data_dir, satellite_str,
                                                                    'n4', '4', 'T89', 'ver4'], varnames=varnames)

        #self.default_save_standard = BasicStandard()

        # Part 1: specify source files to extract variables

        time_var = TimeVariable(name_or_column_in_file='Epoch', original_unit=u.epoch_tt2000, standard_name='Epoch_posixtime')

        variables_to_extract =  {
            'Epoch': time_var, 
            'Energy_FPDU': Variable(name_or_column_in_file='FPDU_Energy', standard_name='Energy_FPDU',
                     original_unit=u.MeV, target_unit=u.MeV, time_bin_method=TimeBinMethod.NoBinning, time_variable=None),
            'FPDU': Variable(name_or_column_in_file='FPDU', standard_name='FPDU',
                     original_unit='', time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var),
            'PA_local_FPDU': Variable(name_or_column_in_file='PA_Midpoint', original_unit=u.deg, target_unit='', standard_name='PA_local', time_bin_method=TimeBinMethod.NoBinning, time_variable=None),
            'xGEO': Variable(name_or_column_in_file='Position', standard_name='xGEO',
                     original_unit=u.RE, target_unit=u.RE, time_bin_method=TimeBinMethod.NanMedian, time_variable=time_var)
        }

        data_path_stem = f"{download_data_dir}rbsp/{satellite_str}/rbspice/l3/pap_tofxeh/YYYY/"
        file_name_stem = f"rbsp-{satellite_str[-1]}-rbspice_lev-3-pap_tofxeh_YYYYMMDD*.cdf"

        source_file = SourceFile(
            download_url=f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/{satellite_str}/l3/rbspice/pap_tofxeh/YYYY/",
            download_arguments_prefixes=f"-r -np -N -nH -e robots=off --cut-dirs=8 --accept '{file_name_stem}'",
            download_arguments_suffixes=f"-P {data_path_stem}",
            download_path=f"{data_path_stem}{file_name_stem}",
            variables_to_extract=variables_to_extract
        )

        self.source_files = [source_file]

        # Part 2: time binning
        self.perform_time_binning = True
        self.time_binning_cadence = timedelta(minutes=5)

        # Part 3: specify derived variables
        self.derived_variables = {
            'PSD_FPDU': DerivedVariable(standard_name='PSD_FPDU', target_unit='', time_variable=time_var),
            'B_local_T89': DerivedVariable(standard_name=f'B_local_T89', target_unit=u.nT, time_variable=time_var),
            'MLT_T89': DerivedVariable(standard_name=f'MLT_T89', target_unit=u.hour, time_variable=time_var),
            'R_eq_T89': DerivedVariable(standard_name=f'R_eq_T89', target_unit=u.RE, time_variable=time_var),
            'B_eq_T89': DerivedVariable(standard_name=f'B_eq_T89', target_unit=u.nT, time_variable=time_var),
            #'Lstar_T89': DerivedVariable(standard_name=f'Lstar_T89', target_unit='', time_variable=time_var),
            'PA_eq_T89': DerivedVariable(standard_name=f'PA_eq_T89', target_unit='', time_variable=time_var),
            'invMu_T89': DerivedVariable(standard_name=f'invMu_T89', target_unit=u.MeV*u.G, time_variable=time_var),
            'invK_T89': DerivedVariable(standard_name=f'invK_T89', target_unit=u.RE*u.G**0.5, time_variable=time_var)
        }

    def process_original_file(self) -> None:
        """Processes the variables from the original files for the product."""

        Flux_arr = self.extracted_variables['FPDU'].data_content
        Flux_arr[Flux_arr <= 0] = np.nan
        self.extracted_variables['FPDU'].data_content = Flux_arr

    def post_process(self, fieldmodel: str = "T89") -> None:
        """Post-processes the product data."""

        # compute_mu_product(self, 'proton', 'Energy_FPDU', 'alpha_eq',
        #                    'B_eq', 'Epoch', 'PA_local_FPDU', fieldmodel)
        # compute_psd_product(self, 'proton', 'FPDU', 'Energy_FPDU', 'Epoch', 'PA_local_FPDU')

    def process_after_binning(self) -> None:
        """Processes the product data after time binning."""
        fold_pitch_angles_and_flux(self)