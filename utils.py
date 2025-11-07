from langchain_core.documents import Document
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
 

LOG_DIR = "logs"
LOGFILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOGFILE = os.path.join(LOG_DIR, "error.log")


def prefix_passage_texts(split_docs, filename):
    prefixed_docs = []
    for doc in split_docs:
        new_doc = Document(
            page_content="passage: " + doc.page_content.strip(),
            metadata=doc.metadata
        )
        new_doc.metadata["filename"] = filename
        prefixed_docs.append(new_doc)
    return prefixed_docs


# === Log file paths ===


def setup_logger(name: str, log_file: str, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Rotating file handler (5MB per file, 5 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)


    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


# === Initialize loggers ===
app_logger = setup_logger("app_logger", LOGFILE, level=logging.INFO)
error_logger = setup_logger("error_logger", ERROR_LOGFILE, level=logging.ERROR)


# === Unified logging helper ===
def write_log(message: str, level: str = "info", exc: Exception = None):
    if exc:
        error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        full_message = f"{message}\n{error_details}"
    else:
        full_message = message
    
    if level.lower() == "error":
        error_logger.error(full_message)
    else:
        app_logger.info(full_message)
    # this means that any thing logged by the app_logger will also be logged by the error logger since it is a lower value?
        

