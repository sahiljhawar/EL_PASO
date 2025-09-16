<!--
SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Bernhard Haas

SPDX-License-Identifier: Apache-2.0
-->

EL-PASO defines custom [astropy units](https://docs.astropy.org/en/stable/units/index.html) for unit conversion.
This is useful for converting between different time formats, as tt2000 and posixtime.

| Unit name | Description |
| :--- | :--- |
| **el_paso.units.cdf_epoch** | Time unit representing milliseconds since 0000-01-01. |
| **el_paso.units.tt2000** | Time unit representing nanoseconds since 2000-01-01. |
| **el_paso.units.posixtime** | Time unit representing seconds since 1970-01-01. |
| **el_paso.units.datenum** | Time unit representing days since 0000-01-01, as used in MATLAB. |
| **el_paso.units.RE** | Unit of distance representing Earth radii. |

There are eqivalences enabled for all custom units, meaning that conversion works as simple as:

```python
from astropy import units as u
import el_paso as ep

posixtime_q = u.Quantity(1362265200.0, ep.units.posixtime)
cdf_tt2000_q = posixtime_q.to(ep.units.cdf_tt2000)
```
