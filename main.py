import logging
from datetime import timedelta
from datetime import datetime, timezone
from examples.products.GOES_realtime import process_goes_real_time
from examples.products.Arase_realtime import process_arase_xep_real_time, process_arase_pew_real_time
from pathlib import Path
import time
logging.captureWarnings(True)
# cron runs at 15th minute of every hour
start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
end_time = start_time + timedelta(days=1)


log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / Path(f"{start_time.strftime('%Y%m%d%H%M00')}.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%H:%M:%S"


file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))


logger.addHandler(file_handler)
logger.addHandler(console_handler)


logging.info("Starting GOES real-time processing")
logging.info(f"Start time: {start_time}")
logging.info(f"End time: {end_time}")

cores = 36
t1 = time.time()
for i in ["primary", "secondary"]:
    process_goes_real_time(
        satellite_str=i,
        download_data_dir="data/",
        irbem_lib_path="IRBEM/libirbem.so",
        start_time=start_time,
        end_time=end_time,
        num_cores=cores,
    )

logging.info(f"Time taken for GOES: {time.time() - t1} seconds on {cores} cores")


start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
end_time = start_time + timedelta(days=1) - timedelta(minutes=1)

logging.info("Starting Arase XEP real-time processing")
logging.info(f"Start time: {start_time}")
logging.info(f"End time: {end_time}")

t2 = time.time()
process_arase_xep_real_time(
    download_data_dir="data/",
    irbem_lib_path="IRBEM/libirbem.so",
    start_time=start_time,
    end_time=end_time,
    num_cores=cores,
)

logging.info(f"Time taken for Arase: {time.time() - t2} seconds on {cores} cores")


# start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
# end_time = start_time + timedelta(days=1) - timedelta(minutes=1)

# logging.info("Starting Arase PEW real-time processing")
# logging.info(f"Start time: {start_time}")
# logging.info(f"End time: {end_time}")

# t2 = time.time()
# process_arase_pew_real_time(
#     download_data_dir="data/",
#     save_data_dir=".",
#     irbem_lib_path="IRBEM/libirbem.so",
#     start_time=start_time,
#     end_time=end_time,
#     num_cores=cores,
# )

# logging.info(f"Time taken for Arase: {time.time() - t2} seconds on {cores} cores")