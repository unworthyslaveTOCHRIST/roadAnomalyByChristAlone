# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

import os
import django
import time
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Graciously setting up Django environment around script
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_for_christ.settings")
django.setup()

from RoadAnomalyManualDataCollection.models import RoadAnomalyManualDataCollection
from RoadAnomalyInput.models import RoadAnomalyInput
from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs
from RoadAnomalyInferenceRaw.models import RoadAnomalyInferenceRaw
from RoadAnomalyPredictionOutput.models import RoadAnomalyPredictionOutput
from RoadAnomalyVerification.models import RoadAnomalyVerification

MODEL_PATH = "road_anomaly_model.pkl"

CSV_BACKUP_DIR = "csv_backups"
os.makedirs(CSV_BACKUP_DIR, exist_ok = True)

# Utility: Export model to CSV (avoiding duplicates)
def export_to_csv(django_model, filename):
    df = pd.DataFrame(django_model.objects.all().values())
    if df.empty:
        return

    full_path = os.path.join(CSV_BACKUP_DIR, filename)


    # If file exists, read it and append only new rows (by ID)
    if os.path.exists(full_path):
        existing_df = pd.read_csv(full_path)
        new_rows = df[~df["id"].isin(existing_df["id"])]
        if not new_rows.empty:
            new_rows.to_csv(full_path, mode = "a", header = False, index = False)

    else:
        df.to_csv(full_path, index = False)


# Extracting Training data
def fetch_training_data():
    data = RoadAnomalyManualDataCollection.objects.all().values()
    df = pd.DataFrame(data)
    if df.empty:
        return None, None

    features = df[["acc_x","acc_y","acc_z","rot_x", "rot_y", "rot_z", "speed"]]
    labels = df["anomaly"]
    return features, labels


def train_and_evaluate():
    X,y = fetch_training_data()
    if X is None:
        print("No data available for training.")
        return

    X_train,X_val,y_train,y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
    clf = RandomForestClassifier(n_estimators=100, random_state = 42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict = True)
    print("Classification report:", report)
    model_performance_metrics = report["weighted avg"]
    print("Model's Performance Metrics:", model_performance_metrics)
    f1_score = model_performance_metrics["f1-score"]

    if f1_score >= 0.90:
        joblib.dump(clf, MODEL_PATH)
        print("✅ New model saved.")
    else:
        print("⚠️ F1-Score not sufficient. Model not updated.")

def update_input_main():
    data = RoadAnomalyManualDataCollection.objects.all().values("latitude", "longitude", "anomaly")
    df = pd.DataFrame(data)
    if df.empty:
        return


    grouped = df.groupby(["latitude", "longitude"]).agg(lambda x : x.mode()[0]).reset_index()
    for _,row in grouped.iterrows():
        RoadAnomalyInput.objects.get_or_create(
            latitude = row.latitude,
            longitude = row.longitude,
            anomaly = row.anomaly

        )

def run_predictions():
    if not os.path.exists(MODEL_PATH):
        print("🚫 Model file not found.")
        return

    clf = joblib.load(MODEL_PATH)



    last_entry = RoadAnomalyInferenceRaw.objects.last
    if last_entry:
        lines = last_entry.data.strip().split("\n")

        for line in lines:
            fields = line.strip().split(",")

            if len(fields) < 12:
                continue # Graciously skipping invalid lines

            try:
                RoadAnomalyInferenceLogs.objects.create(
                    batch_id = int(fields[0]),
                    acc_x = float(fields[1]),
                    acc_y = float(fields[2]),
                    acc_z = float(fields[3]),
                    rot_x = float(fields[4]),
                    rot_y = float(fields[5]),
                    rot_z = float(fields[6]),
                    speed = float(fields[7]),
                    latitude = float(fields[7]),
                    longitude = float(fields[8]),
                    accuracy = float(fields[9]),
                    timestamp = fields[10],
                    log_interval = int(fields[11]),
                )

            except Exception as e:
                print(f"Graciously skipped row due to error: {e}")

    inf_count = RoadAnomalyInferenceLogs.objects.count()
    if inf_count >= 60:
        inf_count_off_start = inf_count - 60
        inf_count_off_end = inf_count
    else :
        inf_count_off_start = 0
        inf_count_off_end = inf_count


    logs = RoadAnomalyInferenceLogs.objects.order_by("id")[inf_count_off_start:inf_count_off_end].values()

    df = pd.DataFrame(logs)
    if df.empty:
        print("No logs to predict")
        return

    features = df[["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "speed"]] # Graciously getting all rows
    predictions = clf.predict(features)

    RoadAnomalyPredictionOutput.objects.all().delete() # Emptying away all previous predictions

    # BY GOD'S GRACE ALONE, Counting occurences of each class
    prediction_series = pd.Series(predictions)
    prediction_counts = prediction_series.value_counts(normalize=True) * 100 #Graciously converting to percentages

    # BY GOD'S GRACE ALONE, using the most recent batch_id (or choosing majority if mixed)
    recent_batch_id = df["batch_id"].mode()[0] if "batch_id" in df else None
    print("Graciously defined batch_id:", recent_batch_id)

    for anomaly_class, percentage in prediction_counts.items():
        RoadAnomalyPredictionOutput.objects.create(
            batch_id = recent_batch_id,
            acc_x=None,
            acc_y=None,
            acc_z=None,
            rot_x=None,
            rot_y=None,
            rot_z=None,
            speed=None,
            latitude=None,
            longitude=None,
            accuracy=None,
            timestamp=str(datetime.now()),
            log_interval=None,
            anomaly_prediction=f"{anomaly_class} ({percentage:.2f}%)"
        )

    print("Aggregated Predictions made.")
    RoadAnomalyInferenceLogs.objects.all().delete()


def process_verification():
    verifications = RoadAnomalyVerification.objects.all()
    for v in verifications:
        if v.response == "YES":
            RoadAnomalyManualDataCollection.objects.create(
                acc_x=v.acc_x,
                acc_y=v.acc_y,
                acc_z=v.acc_z,
                rot_x=v.rot_x,
                rot_y=v.rot_y,
                rot_z=v.rot_z,
                speed=v.speed,
                latitude=v.latitude,
                longitude=v.longitude,
                accuracy=v.accuracy,
                timestamp=v.timestamp,
                anomaly=v.anomaly_prediction
            )

            v.delete()


        else:
            pass




if __name__ == "__main__":
    print("⏳ Running periodic training and prediction cycle with backups...")

    while True:
        train_and_evaluate()
        update_input_main()
        run_predictions()
        process_verification()

        # Graciously saving models to CSV (once every cycle)
        export_to_csv(RoadAnomalyManualDataCollection, "manual_data.csv")
        export_to_csv(RoadAnomalyInput, "input_processing.csv")
        time.sleep(300)  #Graciously pause program execution for 5 minutes







