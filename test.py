from datetime import datetime, timedelta, timezone

from examples.products.GOES_realtime import process_goes_real_time
from examples.products.VanAllenProbes import process_rbsp_hope_electron
from examples.products.DSX import process_DSX_orbit

# process_rbsp_hope_electron('rbspa', './', './', 'IRBEM/libirbem.so', datetime(2017,10,1), datetime(2017,10,2,23,59), num_cores=64)

# start_time = datetime.now(tz=timezone.utc)
# start_time = start_time.replace(hour=0, minute=0, second=0)

# process_goes_real_time(
#     "secondary", "./", "./", "IRBEM/libirbem.so", start_time, datetime.now(tz=timezone.utc), num_cores=64
# )

process_DSX_orbit('./', './', 'IRBEM/libirbem.so', datetime(2019,7,30,17,4, tzinfo=timezone.utc), datetime(2020,4,22,7,19, tzinfo=timezone.utc))