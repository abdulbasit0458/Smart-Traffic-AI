import csv
from datetime import datetime
import os

FILE_NAME = "plates_log.csv"

# CSV create only once
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "time"])


def save_plate(text):
    text = text.strip().upper()

    if len(text) < 5:
        return

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(FILE_NAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([text, time_now])