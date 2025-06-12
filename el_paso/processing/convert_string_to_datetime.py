from datetime import timezone

import numpy as np
from dateutil import parser

from el_paso import Variable


def convert_string_to_datetime(time_var:Variable):

    time_var.metadata.add_processing_note("Converting string-time to datetime")

    return np.asarray([parser.parse(t).replace(tzinfo=timezone.utc) for t in time_var.get_data()])
