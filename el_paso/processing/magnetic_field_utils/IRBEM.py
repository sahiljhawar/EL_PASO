# SPDX-FileCopyrightText: 2022 Mykhaylo Shumko
# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: LGPL-3.0-only

import copy
import ctypes
import os
import pathlib
import shutil
import sys
import typing
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple

import dateutil.parser
import numpy as np
import scipy.interpolate
from numpy.typing import NDArray

from el_paso.processing.magnetic_field_utils.construct_maginput import MagInputKeys

__author__ = "Mykhaylo Shumko"
__last_modified__ = "2022-06-16"
__credit__ = "IRBEM-LIB development team"

"""
Copyright 2022, Mykhaylo Shumko

IRBEM magnetic coordinates and fields wrapper class for Python. Source code
credit goes to the IRBEM-LIB development team.

***************************************************************************
IRBEM-LIB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

IRBEM-LIB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with IRBEM-LIB.  If not, see <http://www.gnu.org/licenses/>.
***************************************************************************
"""
try:
    import pandas as pd

    pandas_imported = True
except ModuleNotFoundError as err:
    if str(err) == "No module named 'pandas'":
        pandas_imported = False
    else:
        raise

# Physical constants
Re = 6371  # km
c = 3.0e8  # m/s

# External magnetic field model look up table.
EXT_MODELS = [
    "None",
    "MF75",
    "TS87",
    "TL87",
    "T89",
    "OPQ77",
    "OPD88",
    "T96",
    "OM97",
    "T01",
    "T01S",
    "T04",
    "A00",
    "T07",
    "MT",
]


class MakeLstarOutput(NamedTuple):
    lm: NDArray[np.float64]
    mlt: NDArray[np.float64]
    blocal: NDArray[np.float64]
    bmin: NDArray[np.float64]
    lstar: NDArray[np.float64]
    xj: NDArray[np.float64]


class GetFieldMultiOutput(NamedTuple):
    bgeo: NDArray[np.float64]
    blocal: NDArray[np.float64]


class MakeLstarShellSplittingOutput(NamedTuple):
    lm: NDArray[np.float64]
    mlt: NDArray[np.float64]
    blocal: NDArray[np.float64]
    bmin: NDArray[np.float64]
    lstar: NDArray[np.float64]
    xj: NDArray[np.float64]


class DriftShellOutput(NamedTuple):
    lm: float
    lstar: float
    blocal: NDArray[np.float64]
    bmin: float
    xj: float
    posit: NDArray[np.float64]
    nposit: NDArray[np.int32]


class FindMagEquatorOutput(NamedTuple):
    xgeo: NDArray[np.float64]
    bmin: float


class TraceFieldLineOutput(NamedTuple):
    posit: NDArray[np.float64]
    n_posit: int
    lm: float
    blocal: NDArray[np.float64]
    bmin: float
    xj: float

class FindFootPointOutput(NamedTuple):
    x_foot: NDArray[np.float64]
    b_foot: NDArray[np.float64]
    b_foot_mag: NDArray[np.float64]

class FindMirrorPointOutput(NamedTuple):
    blocal: float
    bmin: float
    posit: NDArray[np.float64]

class DriftBounceOrbitOutput(NamedTuple):
    xgeo: NDArray[np.float64]
    bl: NDArray[np.float64]
    l: NDArray[np.float64]
    mlt: NDArray[np.float64]
    b_eq: NDArray[np.float64]
    x_eq: NDArray[np.float64]
    s: NDArray[np.float64]
    alpha: NDArray[np.float64]
    lam: NDArray[np.float64]
    t: NDArray[np.float64]
    npts: NDArray[np.int32]


class MagFields:
    """Wrappers for IRBEM's magnetic field functions.

    Functions wrapped and not tested:
    None at this time

    Special functions not in normal IRBEM (no online documentation yet):
    bounce_period()
    mirror_point_altitude()

    Please contact me at msshumko at gmail.com if you have questions/comments
    or you would like me to wrap a particular function.
    """

    def __init__(
        self,
        *,
        lib_path: str | Path,
        verbose: bool = False,
        kext: int | str = 5,
        sysaxes: int = 0,
        options: Sequence[int] | None = None,
    ):
        """Initialize the MagFields class.

        When initializing the IRBEM instance, you can provide the path kwarg that
        specifies the location of the compiled FORTRAN shared object (.so or .dll)
        file, otherwise, it will search for the shared object file in the top-level
        IRBEM directory. Python wrapper error value is -9999 (IRBEM-Lib's Fortan error value is -1E31).

        Parameters
        ----------
        path: str or pathlib.Path
            An optional path to the IRBEM shared object (.so or .dll). If unspecified, it
            it will search for the shared object file in the top-level IRBEM directory.
        options: list
            array(5) of long integer to set some control options on computed values. See the
            HTML documentation for more information
        kext: str
            The external magnetic field model, defaults to OPQ77.
        sysaxes: str
            Set the input coordinate system. By default set to GDZ (alt, lat, long).
        verbose: bool
            Prints a statement prior to running each function. Usefull for debugging in
            case Python quietly crashes (likely a wrapper or a Fortran issue).
        """
        self.irbem_obj_path = Path(lib_path)
        self.verbose = verbose

        self._irbem_obj = _load_shared_object(self.irbem_obj_path)

        # global model parameters, default is OPQ77 model with GDZ coordinate
        # system. If kext is a string, find the corresponding integer value.
        if isinstance(kext, str):
            try:
                self.kext = ctypes.c_int(EXT_MODELS.index(kext))
            except ValueError as err:
                msg = (
                    "Incorrect external model selected. Valid models are 'None', 'MF75',"
                    "'TS87', 'TL87', 'T89', 'OPQ77', 'OPD88', 'T96', 'OM97'"
                    "'T01', 'T04', 'A00'"
                )
                raise ValueError(msg) from err
        else:
            self.kext = ctypes.c_int(kext)

        self.sysaxes = ctypes.c_int(sysaxes)

        # If options are not supplied, assume they are all 0's.
        options_type = ctypes.c_int * 5
        if options is None:
            self.options = options_type(0, 0, 0, 0, 0)
        else:
            self.options = options_type()
            for i in range(5):
                self.options[i] = options[i]

        # Get the NTIME_MAX value
        self.ntime_max = ctypes.c_int(-1)
        self._irbem_obj.get_irbem_ntime_max1_(ctypes.byref(self.ntime_max))

    def make_lstar(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> MakeLstarOutput:
        # Convert the satellite time and position into c objects.
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        # Convert the model parameters into c objects.
        c_maginput = self._prep_maginput(maginput)

        # Model outputs
        double_arr_type = ctypes.c_double * c_ntime.value
        c_lm, c_lstar, c_blocal, c_bmin, c_xj, c_mlt = [double_arr_type() for i in range(6)]

        if self.verbose:
            print("Running IRBEM-LIB make_lstar")

        self._irbem_obj.make_lstar1_(
            ctypes.byref(c_ntime),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_lm),
            ctypes.byref(c_lstar),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_mlt),
        )

        return MakeLstarOutput(c_lm.value, c_lstar.value, c_blocal.value, c_bmin.value, c_lstar.value, c_xj.value)

    def make_lstar_shell_splitting(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        alpha: Sequence[np.floating] | NDArray[np.floating] | np.floating,
    ) -> MakeLstarShellSplittingOutput:
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        if not isinstance(alpha, Sequence | np.ndarray):
            alpha = [alpha]

        # Cast additional inputs
        c_n_alpha = ctypes.c_int(len(alpha))
        c_alpha = (ctypes.c_double * c_n_alpha.value)()

        for da in range(c_n_alpha.value):
            c_alpha[da] = alpha[da]

        # Convert the model parameters into c objects.
        c_maginput = self._prep_maginput(maginput)

        # Model outputs
        double_arr_type = ctypes.c_double * (c_ntime.value * c_n_alpha.value)
        c_lm, c_lstar, c_blocal, c_bmin, c_xj, c_mlt = [double_arr_type() for _ in range(6)]

        if self.verbose:
            print("Running IRBEM-LIB make_lstar_shell_splitting")

        self._irbem_obj.make_lstar_shell_splitting1_(
            ctypes.byref(c_ntime),
            ctypes.byref(c_n_alpha),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_alpha),
            ctypes.byref(c_maginput),
            ctypes.byref(c_lm),
            ctypes.byref(c_lstar),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_mlt),
        )

        return MakeLstarShellSplittingOutput(
            lm=np.array(c_lm).reshape(c_n_alpha.value, c_ntime.value),
            mlt=np.array(c_mlt).reshape(c_n_alpha.value, c_ntime.value),
            blocal=np.array(c_blocal).reshape(c_n_alpha.value, c_ntime.value),
            bmin=np.array(c_bmin).reshape(c_n_alpha.value, c_ntime.value),
            lstar=np.array(c_lstar).reshape(c_n_alpha.value, c_ntime.value),
            xj=np.array(c_xj).reshape(c_n_alpha.value, c_ntime.value),
        )

    def drift_shell(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> DriftShellOutput:
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        if self.verbose:
            print("Running IRBEM-LIB drift_shell for multiple time steps")

        c_posit = (((ctypes.c_double * 3) * 1000) * 48)()
        c_nposit = (48 * ctypes.c_int)()
        c_lm, c_lstar, c_bmin, c_xj = [ctypes.c_double() for _ in range(4)]
        c_blocal = ((ctypes.c_double * 1000) * 48)()

        self._irbem_obj.drift_shell1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_lm),
            ctypes.byref(c_lstar),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_posit),
            ctypes.byref(c_nposit),
        )

        posit = np.array(c_posit)
        nposit = np.array(c_nposit)
        for i, n in enumerate(nposit):
            posit[i, n:, :] = np.nan

        return DriftShellOutput(
            lm=c_lm.value,
            lstar=c_lstar.value,
            blocal=np.array(c_blocal),
            bmin=c_bmin.value,
            xj=c_xj.value,
            posit=posit,
            nposit=nposit,
        )

    def find_mirror_point(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        alpha: float,
    ) -> FindMirrorPointOutput:
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_alpha = ctypes.c_double(alpha)

        if self.verbose:
            print("Running IRBEM-LIB find_mirror_point for multiple time steps and pitch angles")

        c_blocal = c_bmin = ctypes.c_double(-9999)
        c_posit = (3 * ctypes.c_double)()

        self._irbem_obj.find_mirror_point1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_alpha),
            ctypes.byref(c_maginput),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_posit),
        )

        return FindMirrorPointOutput(
            blocal=c_blocal.value,
            bmin=c_bmin.value,
            posit=np.array(c_posit),
        )

    def find_foot_point(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        stop_alt: float,
        hemi_flag: int,
    ) -> FindFootPointOutput:
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_stop_alt = ctypes.c_double(stop_alt)
        c_hemi_flag = ctypes.c_int(hemi_flag)

        # Output lists for multiple time steps
        xfoot_list: list[np.ndarray] = []
        bfoot_list: list[np.ndarray] = []
        bfootmag_list: list[float] = []

        if self.verbose:
            print("Running IRBEM-LIB find_foot_point")

        # The underlying Fortran function is not vectorized over time, so we must loop
        for i in range(c_ntime.value):
            # Define output variables for a single time step
            c_xfoot = (ctypes.c_double * 3)()
            c_bfoot = (ctypes.c_double * 3)()
            c_bfootmag = ctypes.c_double()

            # Get single-time-step inputs
            c_iyear_i = ctypes.c_int(c_iyear[i])
            c_idoy_i = ctypes.c_int(c_idoy[i])
            c_ut_i = ctypes.c_double(c_ut[i])
            c_position_i = (ctypes.c_double * 3)(c_x1[i], c_x2[i], c_x3[i])

            self._irbem_obj.find_foot_point1_(
                ctypes.byref(self.kext),
                ctypes.byref(self.options),
                ctypes.byref(self.sysaxes),
                ctypes.byref(c_iyear_i),
                ctypes.byref(c_idoy_i),
                ctypes.byref(c_ut_i),
                ctypes.byref(c_position_i),
                ctypes.byref(c_stop_alt),
                ctypes.byref(c_hemi_flag),
                ctypes.byref(c_maginput),
                ctypes.byref(c_xfoot),
                ctypes.byref(c_bfoot),
                ctypes.byref(c_bfootmag),
            )

            xfoot_list.append(np.array(c_xfoot))
            bfoot_list.append(np.array(c_bfoot))
            bfootmag_list.append(c_bfootmag.value)

        # Stack the results into a single NumPy array, adding a new dimension for time
        return FindFootPointOutput(
            x_foot=np.array(xfoot_list),
            b_foot=np.array(bfoot_list),
            b_foot_mag=np.array(bfootmag_list),
        )

    def trace_field_line(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
        r0: float = 1,
    ) -> TraceFieldLineOutput:

        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        c_r0 = ctypes.c_double(r0)

        if self.verbose:
            print("Running IRBEM-LIB trace_field_line for multiple time steps")

        c_posit = ((ctypes.c_double * 3) * 3000)()
        c_n_posit = ctypes.c_int(-9999)
        c_lm = c_blocal = c_bmin = c_xj = ctypes.c_double(-9999)
        c_blocal = (ctypes.c_double * 3000)()

        self._irbem_obj.trace_field_line1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_r0),
            ctypes.byref(c_lm),
            ctypes.byref(c_blocal),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xj),
            ctypes.byref(c_posit),
            ctypes.byref(c_n_posit),
        )

        return TraceFieldLineOutput(
            posit = np.array(c_posit[: c_n_posit.value]),
            n_posit = c_posit.value,
            lm = c_lm.value,
            blocal = np.array(c_blocal[: c_n_posit.value]),
            bmin = c_bmin.value,
            xj = c_xj.value,
        )

    def find_magequator(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> FindMagEquatorOutput:

        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)
        c_maginput = self._prep_maginput(maginput)

        if self.verbose:
            print("Running IRBEM-LIB find_magequator for multiple time steps")

        c_xgeo = (ctypes.c_double * 3)(-9999, -9999, -9999)
        c_bmin = ctypes.c_double(-9999)

        self._irbem_obj.find_magequator1_(
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_bmin),
            ctypes.byref(c_xgeo),
        )

        return FindMagEquatorOutput(xgeo=np.array(c_xgeo), bmin=c_bmin.value)

    def get_field_multi(
        self,
        time: Sequence[datetime | str] | datetime | str,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
        maginput: Mapping[MagInputKeys, NDArray[np.number] | list[np.number] | np.number],
    ) -> GetFieldMultiOutput:
        # Prep the time and position variables.
        c_ntime, c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos_array(time, position)

        # Prep magnetic field model inputs
        c_maginput = self._prep_maginput(maginput)

        # Model output arrays
        c_bmag = (ctypes.c_double * c_ntime.value)()
        c_bgeo = ((ctypes.c_double * 3) * c_ntime.value)()

        if self.verbose:
            print("Running IRBEM-LIB get_field_multi")

        self._irbem_obj.get_field_multi_(
            ctypes.byref(c_ntime),
            ctypes.byref(self.kext),
            ctypes.byref(self.options),
            ctypes.byref(self.sysaxes),
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_x1),
            ctypes.byref(c_x2),
            ctypes.byref(c_x3),
            ctypes.byref(c_maginput),
            ctypes.byref(c_bgeo),
            ctypes.byref(c_bmag),
        )

        return GetFieldMultiOutput(np.array(c_bgeo), np.array(c_bmag))

    def get_mlt(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
    ) -> float:
        c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3 = self._prep_time_pos(time, position)

        if self.verbose:
            print("Running IRBEM-LIB get_mlt in a time loop")

        c_mlt = ctypes.c_double(-9999)
        c_position = (ctypes.c_double * 3)(c_x1, c_x2, c_x3)

        self._irbem_obj.get_mlt1_(
            ctypes.byref(c_iyear),
            ctypes.byref(c_idoy),
            ctypes.byref(c_ut),
            ctypes.byref(c_position),
            ctypes.byref(c_mlt),
        )

        return c_mlt.value

    def _prep_time_pos(
        self,
        time: datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], np.floating],
    ) -> tuple[ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]:
        if self.verbose:
            print("Prepping time and space input variables")

        # Deep copy X so if the single inputs get encapsulated in
        # an array, it wont be propaged back to the user.
        position = copy.deepcopy(position)
        time = copy.deepcopy(time)

        if isinstance(time, datetime):
            time_dt = time
        elif isinstance(time, pd.Timestamp):
            time_dt = time.to_pydatetime()
        else:
            time_dt = dateutil.parser.parse(time)

        c_iyear = ctypes.c_int(time_dt.year)
        c_idoy = ctypes.c_int(time_dt.timetuple().tm_yday)
        c_ut = ctypes.c_double(3600 * time_dt.hour + 60 * time_dt.minute + time_dt.second)  # Seconds of day
        c_x1 = ctypes.c_double(float(position["x1"]))
        c_x2 = ctypes.c_double(float(position["x2"]))
        c_x3 = ctypes.c_double(float(position["x3"]))

        if self.verbose:
            print("Done prepping time and space input variables")

        return c_iyear, c_idoy, c_ut, c_x1, c_x2, c_x3

    def _prep_time_pos_array(
        self,
        time: Sequence[datetime | str | pd.Timestamp] | NDArray[np.generic] | datetime | str | pd.Timestamp,
        position: Mapping[Literal["x1", "x2", "x3"], Sequence[np.floating] | NDArray[np.floating] | np.floating],
    ) -> tuple[
        ctypes.c_int,
        ctypes.Array[ctypes.c_int],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
        ctypes.Array[ctypes.c_double],
    ]:
        time = copy.deepcopy(time)
        time = np.atleast_1d(np.asarray(time))

        position = dict(copy.deepcopy(position))
        position["x1"] = np.atleast_1d(np.asarray(position["x1"]))
        position["x2"] = np.atleast_1d(np.asarray(position["x2"]))
        position["x3"] = np.atleast_1d(np.asarray(position["x3"]))

        # Check that the input array length does not exceed NTIME_MAX.
        if len(time) > self.ntime_max.value:
            msg = (
                f"Input array length {len(time)} is longer "
                f"than IRBEM's NTIME_MAX = {self.ntime_max.value}. "
                f"Use a for loop."
            )
            raise ValueError(msg)
        ntime = ctypes.c_int(len(time))

        # Check that the times are datetime objects, and convert otherwise.
        if isinstance(time[0], datetime):
            time = typing.cast("Sequence[datetime]", time)
            time_dt = time
        elif isinstance(time[0], pd.Timestamp):
            time = typing.cast("Sequence[pd.Timestamp]", time)
            time_dt = [t.to_pydatetime() for t in time]
        else:
            time = typing.cast("Sequence[str]", time)
            time_dt = [dateutil.parser.parse(t) for t in time]

        # C arrays are statically defined with the following procedure.
        # There are a few ways of doing this...
        int_arr_type = ctypes.c_int * len(time_dt)
        iyear = int_arr_type()
        idoy = int_arr_type()

        double_arr_type = ctypes.c_double * len(time_dt)
        ut, x1, x2, x3 = [double_arr_type() for i in range(4)]

        # Now fill the input time and model sampling (s/c location) parameters.
        for dt in range(len(time_dt)):
            iyear[dt] = time_dt[dt].year
            idoy[dt] = time_dt[dt].timetuple().tm_yday
            ut[dt] = 3600 * time_dt[dt].hour + 60 * time_dt[dt].minute + time_dt[dt].second
            x1[dt] = position["x1"][dt]
            x2[dt] = position["x2"][dt]
            x3[dt] = position["x3"][dt]

        return ntime, iyear, idoy, ut, x1, x2, x3

    def _prep_maginput(
        self, input_dict: Mapping[MagInputKeys, list[np.number] | NDArray[np.number] | np.number] | None = None
    ) -> ctypes.Array[ctypes.c_double]:
        if self.verbose:
            print("Prepping magnetic field inputs.")

        # If no model inputs (statis magnetic field model)
        if (input_dict is None) or (input_dict == {}):
            maginput = (ctypes.c_double * 25)()
            for i in range(25):
                maginput[i] = -9999
            return maginput

        ordered_keys: list[MagInputKeys] = [
            "Kp",
            "Dst",
            "dens",
            "velo",
            "Pdyn",
            "ByIMF",
            "BzIMF",
            "G1",
            "G2",
            "G3",
            "W1",
            "W2",
            "W3",
            "W4",
            "W5",
            "W6",
            "AL",
        ]

        # If the model inputs are arrays
        if isinstance(next(iter(input_dict.values())), list | np.ndarray):
            input_dict = typing.cast("Mapping[MagInputKeys, list[np.number] | NDArray[np.number]]", input_dict)
            ntime = len(next(iter(input_dict.values())))

            maginput = ((ctypes.c_double * 25) * ntime)()

            for i, key in enumerate(ordered_keys):
                for dt in range(ntime):
                    if key in input_dict:
                        maginput[dt][i] = input_dict[key][dt]
                    else:
                        maginput[dt][i] = ctypes.c_double(-9999)

        else:
            maginput = (ctypes.c_double * 25)()

            for i, key in enumerate(ordered_keys):
                if key in input_dict:
                    maginput[i] = input_dict[key]
                else:
                    maginput[i] = ctypes.c_double(-9999)

        if self.verbose:
            print("Done prepping magnetic field inputs.")

        return maginput

class Coords:
    """Wrappers for IRBEM's coordinate transformation functions.

    When initializing the instance, you can provide the directory
    'IRBEMdir' and 'IRBEMname' arguments to the class to specify the location
    of the  compiled FORTRAN shared object (so) file, otherwise, it will
    search for a .so file in the ./../ directory.

    When creating the instance object, you can use the 'options' kwarg to
    set the options, dafault is 0,0,0,0,0. Kwarg 'kext' sets the external B
    field as is set to default of 4 (T89c model), and 'sysaxes' kwarg sets the
    input coordinate system, and is set to GDZ (lat, long, alt).

    verbose keyword, set to False by default, will print too much information
    (TMI). Usefull for debugging and for knowing too much. Set it to True if
    Python quietly crashes (probably an input to Fortran issue)

    Python wrapper error value is -9999.

    TESTING IRBEM: Run coords_tests_and_visalization.py (FORTRAN coord_trans_vec1)
    Rough validation was done with "Heliospheric Coordinate Systems" by Franz and
    Harper 2017.

    WRAPPED_FUNCTION:
        coords_transform(self, time, pos, sysaxesIn, sysaxesOut)

    Please contact me at msshumko at gmail.com if you have questions/comments
    or you would like me to wrap a particular function.
    """

    def __init__(self, **kwargs):
        self.irbem_obj_path = kwargs.get("path")
        self.verbose = kwargs.get("verbose", False)

        self._irbem_obj = _load_shared_object(self.irbem_obj_path)

    def coords_transform(self, *args, **kwargs):
        warnings.warn("Coords.coords_transform() is deprecated. Use Coords.transform instead.")
        return self.transform(*args, **kwargs)

    def transform(self, time, pos, sysaxesIn, sysaxesOut):
        """NAME:  coords_transform(self, X, sysaxesIn, sysaxesOut)
        USE:   This function transforms coordinate systems from a point at time
               time and position pos from a coordinate system sysaxesIn to
               sysaxesOut.
        INPUT:  time - datetime objects
                       (or arrays/lists containing them)
                pos - A (nT x 3) array where nT is the number of points to transform.

                Avaliable coordinate transformations (either as an integer or
                3 letter keyword will work as arguments)

                0: GDZ (alti, lati, East longi - km,deg.,deg)
                1: GEO (cartesian) - Re
                2: GSM (cartesian) - Re
                3: GSE (cartesian) - Re
                4: SM (cartesian) - Re
                5: GEI (cartesian) - Re
                6: MAG (cartesian) - Re
                7: SPH (geo in spherical) - (radial distance, lati, East
                    longi - Re, deg., deg.)
                8: RLL  (radial distance, lati, East longi - Re, deg.,
                    deg. - prefered to 7)
        AUTHOR: Mykhaylo Shumko
        RETURNS: Transformed positions as a 1d or 2d array.
        MOD:     2017-07-17
        """
        # Create the position arrays
        if hasattr(time, "__len__"):
            pos = np.array(pos)
            pos = pos.reshape((len(time), 3))
            posArrType = (ctypes.c_double * 3) * len(time)
            nTime = ctypes.c_int(len(time))
        else:
            pos = np.array([pos])
            posArrType = (ctypes.c_double * 3) * 1
            nTime = ctypes.c_int(1)
        posInArr = posArrType()
        posOutArr = posArrType()

        ### Get the time entries ###
        iyear, idoy, ut = self._cTimes(time)

        ### Lookup coordinate systems ###
        sysIn = self._coordSys(sysaxesIn)
        sysOut = self._coordSys(sysaxesOut)

        # Fill the positions array.
        for nT in range(pos.shape[0]):
            for nX in range(pos.shape[1]):
                posInArr[nT][nX] = ctypes.c_double(pos[nT, nX])

        self._irbem_obj.coord_trans_vec1_(
            ctypes.byref(nTime),
            ctypes.byref(sysIn),
            ctypes.byref(sysOut),
            ctypes.byref(iyear),
            ctypes.byref(idoy),
            ctypes.byref(ut),
            ctypes.byref(posInArr),
            ctypes.byref(posOutArr),
        )
        return np.array(posOutArr[:])

    def _cTimes(self, times):
        """NAME:  _cTimes(self, times)
        USE:   This is a helper function that takes in an array of times in ISO
                format or datetime format and returns it in ctypes format with
                iyear, idoy, and ut.
        INPUT: times as datetime or ISO string objects. Or an array/list of those
                objects.
        AUTHOR: Mykhaylo Shumko
        RETURNS: Arrays of iyear, idoy, ut.
        MOD:     2017-07-14
        """
        if not hasattr(times, "__len__"):  # Make an array if only one value supplied.
            times = np.array([times])
        N = len(times)

        # Intialize the C arrays
        tArrType = ctypes.c_int * N
        utArrType = ctypes.c_double * N
        iyear, idoy = [tArrType() for i in range(2)]
        ut = utArrType()

        # Convert to datetimes if necessary.
        if isinstance(times[0], str):
            t = list(map(dateutil.parser.parse, times))
        elif isinstance(times[0], datetime):
            t = times
        else:
            raise ValueError(
                "Unknown time format. Valid formats: ISO string, datetime objects, or arrays of those objects"
            )

        for nT in range(N):  # Populate C arrays
            iyear[nT] = ctypes.c_int(t[nT].year)
            idoy[nT] = ctypes.c_int(t[nT].timetuple().tm_yday)
            ut[nT] = ctypes.c_double(3600 * t[nT].hour + 60 * t[nT].minute + t[nT].second)
        return iyear, idoy, ut

    def _coordSys(self, coordSystem):
        """NAME:  _coordSys(self, axes)
        USE:   This function looks up the IRBEM coordinate system integer, given
               an input integer, or string representing the coordinate system.
        INPUT: axes, a coordinate system from:
                0: GDZ (alti, lati, East longi - km,deg.,deg)
                1: GEO (cartesian) - Re
                2: GSM (cartesian) - Re
                3: GSE (cartesian) - Re
                4: SM (cartesian) - Re
                5: GEI (cartesian) - Re
                6: MAG (cartesian) - Re
                7: SPH (geo in spherical) - (radial distance, lati, East
                    longi - Re, deg., deg.)
                8: RLL  (radial distance, lati, East longi - Re, deg.,
                    deg. - prefered to 7)
               either an integer or a 3 letter string.
        AUTHOR: Mykhaylo Shumko
        RETURNS: IRBEM sysaxes integer
        MOD:     2017-07-14
        """
        lookupTable = {"GDZ": 0, "GEO": 1, "GSM": 2, "GSE": 3, "SM": 4, "GEI": 5, "MAG": 6, "SPH": 7, "RLL": 8}

        if isinstance(coordSystem, str):
            assert coordSystem.upper() in lookupTable, (
                "ERROR: Unknown coordinate system! Choose from GDZ, GEO, GSM, GSE, SM, GEI, MAG, SPH, RLL."
            )
            return ctypes.c_int(lookupTable[coordSystem])
        if isinstance(coordSystem, int):
            return ctypes.c_int(coordSystem)
        raise ValueError("Error, coordinate axis can only be a string or int!")


def _load_shared_object(path: Path | None = None) -> ctypes.CDLL:
    """Searches for and loads a shared object (.so or .dll file).

    If path is specified it doesn't search for the file.
    """
    if path is None:
        if (sys.platform == "win32") or (sys.platform == "cygwin"):
            obj_name = "libirbem.dll"
        else:
            obj_name = "libirbem.so"
        matched_object_files = list(pathlib.Path(__file__).parents[2].rglob(obj_name))
        if len(matched_object_files) != 1:
            msg = (
                f"{len(matched_object_files)} .so or .dll shared object files found in "
                f"{pathlib.Path(__file__).parents[2]} folder: {matched_object_files}."
            )
            raise ValueError(msg)

        path = matched_object_files[0]

    # Open the shared object file.
    try:
        if (sys.platform == "win32") or (sys.platform == "cygwin"):
            # Some versions of ctypes (Python) need to know where msys64 binary
            # files are located, or ctypes is unable to load the IREBM dll.
            gfortran_path = pathlib.Path(shutil.which("gfortran.exe"))
            os.add_dll_directory(gfortran_path.parent)  # e.g. C:\msys64\mingw64\bin
            irbem_obj = ctypes.WinDLL(str(path))
        else:
            irbem_obj = ctypes.CDLL(str(path))
    except OSError as err:
        if "Try using the full path with constructor syntax." in str(err):
            msg = f"Could not load the IRBEM shared object file in {path}"
            raise OSError(msg) from err
        raise
    return irbem_obj