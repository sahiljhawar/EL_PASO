from datetime import datetime, timedelta, timezone

from examples.products.GOES_realtime import process_goes_real_time
from examples.products.VanAllenProbes import process_rbsp_hope_electron

# process_rbsp_hope_electron('rbspa', './', './', 'IRBEM/libirbem.so', datetime(2017,10,1), datetime(2017,10,2,23,59), num_cores=64)

start_time = datetime.now(tz=timezone.utc)
start_time = start_time.replace(hour=0, minute=0, second=0)

process_goes_real_time(
    "secondary", "./", "./", "IRBEM/libirbem.so", start_time, datetime.now(tz=timezone.utc), num_cores=64
)
