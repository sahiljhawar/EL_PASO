from datetime import timezone

import numpy as np
from dateutil import parser
from numpy.typing import NDArray

from el_paso import Variable


def convert_string_to_datetime(time_var:Variable) -> NDArray[np.generic]:
    """Converts a Variable's string-based time data to UTC datetime objects.

    Args:
        time_var (Variable): The variable containing string-based time data to be
            converted.

    Returns:
        NDArray[np.generic]: A NumPy array of `datetime` objects in the UTC timezone.
    """
    time_var.metadata.add_processing_note("Converting string-time to datetime")

    return np.asarray([parser.parse(t).replace(tzinfo=timezone.utc) for t in time_var.get_data()])
