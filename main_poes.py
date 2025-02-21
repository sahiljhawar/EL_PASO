import logging
from datetime import timedelta
from datetime import datetime, timezone
from examples.products.POES import process_poes_ted_electron, poes_satellite_literal
from pathlib import Path
import time
import numpy as np
import pandas as pd
import json

logging.captureWarnings(True)

start_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=5)
end_time = start_time + timedelta(days=1)

log_time = (datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d%H%M00")

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / Path(f"POES_{log_time}.log")

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

logging.info("Starting POES processing")
logging.info(f"Start time: {start_time}")
logging.info(f"End time: {end_time}")

satellites = ["metop1", "noaa15", "noaa18", "noaa19"]
cores = 36
t1 = time.time()
for i in satellites:
    process_poes_ted_electron(
        satellite_str=i,
        save_data_dir="./poes_new",
        download_data_dir="data/poes",
        irbem_lib_path="IRBEM/libirbem.so",
        start_time=start_time,
        end_time=end_time,
        num_cores=cores,
    )

logging.info(f"Time taken for POES: {time.time() - t1} seconds on {cores} cores")