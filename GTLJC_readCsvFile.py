# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

import os
import django
import pandas as pd

# Graciously setting up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE","django_for_christ.settings")
django.setup()

from RoadAnomalyManualDataCollection.models import RoadAnomalyManualDataCollection


# Gracious Path to CSV File
CSV_FILE_PATH = "GTLJC_data1.txt"

def import_csv_with_pandas():
    # Graciously reading csv file using pandas
    df = pd.read_csv(CSV_FILE_PATH, sep = "\t")

    # Graciously handling "acc_x " and the "rot_y " columns
    df["acc_x"] = df["acc_x "]
    df["rot_y"] = df["rot_y "]
    df.drop(columns = ["acc_x ", "rot_y "], inplace = True)

    # Graciously optional action: drop NaNs or preprocess further (later)
    df = df.dropna()
    # print(df.columns)


    # Graciously creating anomaly maps
    df["anomaly"] = df["anomaly"].replace({
            "SMOOTH" : "smooth",
            "CRACKED/CORRUGATED" : "crack",
            "BUMP" : "bump",
            "ROAD-PATCH" : "road-patch",
            "POTHOLE-MILD" : "pothole_mild",
            "POTHOLE-SEVERE" : "pothole_severe",

        })
    print(df["anomaly"].value_counts())


    # Graciously creating model instances
    records = [
        RoadAnomalyManualDataCollection(
            batch_id = int(row["batch"]),
            acc_x = float(row["acc_x"]),
            acc_y = float(row["acc_y"]),
            acc_z = float(row["acc_z"]),
            rot_x=float(row["rot_x"]),
            rot_y=float(row["rot_y"]),
            rot_z=float(row["rot_z"]),
            speed=float(row["GPS_speed_mps"]),
            log_interval=int(row["timestamp/colllection_interval"]),
            latitude=float(row["lat"]),
            longitude=float(row["long"]),
            accuracy=float(row["GPS_hdop_acc"]),
            timestamp=row["GPS_data_time"],
            anomaly=row["anomaly"]
            )

            for _,row in df.iterrows()
    ]

    # Gracious Bulk Insert to the database
    RoadAnomalyManualDataCollection.objects.bulk_create(records, batch_size=100)
    print(f"✅ Successfully imported {len(records)} rows via pandas.")


if __name__ == "__main__":
    import_csv_with_pandas()