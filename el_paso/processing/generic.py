from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from el_paso.classes import SourceFile, Variable


def load_variables_from_source_files(
    source_files: SourceFile | Sequence[SourceFile], start_time: datetime, end_time: datetime,
) -> dict[str, Variable]:
    if not isinstance(source_files, Sequence):
        source_files = [source_files]

    variables = {}

    for source_file in source_files:
        variables |= source_file.extract_variables(start_time, end_time)

    return variables


def convert_all_data_to_standard_units(variables: dict[str, Variable]) -> None:
    for var in variables.values():
        var.convert_to_standard_unit()