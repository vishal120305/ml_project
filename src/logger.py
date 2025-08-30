import logging
import os
from datetime import datetime

# Define log directory and file name separately
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s]-%(lineno)d %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_error(error_message):
    logging.error(error_message)

