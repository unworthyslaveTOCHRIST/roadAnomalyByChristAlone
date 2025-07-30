# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

import os
import sys

# Step 1: Add the project root to Python path BEFORE any app imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Step 2: Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_for_christ.settings')  # Adjust as needed

# Step 3: Setup Django
import django
django.setup()

# ✅ Now it's safe to import Django models and app code
import pandas as pd
import joblib

from RoadAnomalyPredictionOutput.utilities import (
    align_to_global_frame,
    fix_batches,
    apply_feature_extraction_across_all_identical_anomaly_batches,
    ml_pipeline
)

from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs


if __name__ == "__main__":
    inference_data = RoadAnomalyInferenceLogs.objects.order_by("id").values()
    df = pd.DataFrame(inference_data)

    if df.empty:
        print("🚫 No inference data found in the database.")
    else:
        data_globally_aligned = align_to_global_frame(df)
        batched_df = fix_batches(data_globally_aligned)
        engineered_df = apply_feature_extraction_across_all_identical_anomaly_batches(batched_df)
        predictions = ml_pipeline(engineered_df)
        print("✅ Predictions:")
        print(predictions)
