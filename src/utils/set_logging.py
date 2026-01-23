import logging

from configs.constants import LOG_DIR
from utils.file_dir import ensure_directory_exists

def get_logger():
    ensure_directory_exists(folder = LOG_DIR)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{LOG_DIR}/log.log', encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    return logger