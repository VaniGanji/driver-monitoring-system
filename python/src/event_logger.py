import csv
import os
from datetime import datetime

LOG_DIR = "results"
LOG_FILE = os.path.join(LOG_DIR, "session_log.csv")

def initialize_logger():
    """
    Create results directory and CSV file if they don't exist.
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Event"])


def log_event(event):
    """
    Append an event with timestamp.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, event])