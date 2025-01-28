from datetime import timezone

import numpy as np
from dateutil import parser

from el_paso.classes import TimeVariable


def convert_string_to_datetime(time_var:TimeVariable):

    time_var.data = np.asarray([parser.parse(t).replace(tzinfo=timezone.utc) for t in time_var.data])
