import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "logs.json")
TRAIN_STATS_FILE = os.path.join(BASE_DIR, "artifacts", "metrics", "train_stats.json")


# ----------------------------------
# LOGGING FUNCTION
# ----------------------------------
def log_prediction(features, prediction, probability):
    """
    Logs each prediction in structured JSON format.
    """

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction": int(prediction),
        "probability": float(probability)
    }

    # Add feature values to log
    for column in features.columns:
        log_entry[column] = float(features.iloc[0][column])

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ----------------------------------
# DRIFT DETECTION FUNCTION
# ----------------------------------
def check_drift(features):
    """
    Checks whether incoming feature values are very different
    from the training data statistics.
    """

    if not os.path.exists(TRAIN_STATS_FILE):
        return []

    with open(TRAIN_STATS_FILE, "r") as f:
        stats = json.load(f)

    drift_flags = []

    for column in features.columns:

        if column not in stats:
            continue

        value = float(features.iloc[0][column])
        mean = stats[column]["mean"]
        std = stats[column]["std"]

        upper_limit = mean + (3 * std)
        lower_limit = mean - (3 * std)

        if value > upper_limit or value < lower_limit:
            drift_flags.append(f"{column} drift")

    return drift_flags