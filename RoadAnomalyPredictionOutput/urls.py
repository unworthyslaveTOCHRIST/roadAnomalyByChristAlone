# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from django.urls import path, include
from rest_framework import serializers, viewsets, routers
from datetime import datetime
from .models import RoadAnomalyPredictionOutput
from rest_framework.response import Response
from rest_framework import status
from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs
from rest_framework.parsers import JSONParser
from .parsers import PlainTextParser

import os
import joblib
import pandas as pd
# import django
# import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .utilities import align_to_global_frame, fix_batches, apply_feature_extraction_across_all_identical_anomaly_batches
from .utilities import ml_pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'RoadAnomalyPredictionOutput', 'road_anomaly_model_2.pkl')


class RoadAnomalyPredictionOutputSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoadAnomalyPredictionOutput
        fields = ("id","batch_id","acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "speed", "timestamp", "log_interval","latitude","longitude","accuracy","anomaly_prediction")


class RoadAnomalyPredictionOutputViewSet(viewsets.ModelViewSet):
    queryset = RoadAnomalyPredictionOutput.objects.all()
    serializer_class = RoadAnomalyPredictionOutputSerializer
    parser_classes = [PlainTextParser, JSONParser]  # <--- Graciously accepts both JSON and plain text --->

    def create(self, request, *args, **kwargs):
        raw_data = request.data.strip()

        if not isinstance(raw_data,str):
            return Response({"error": "Invalid data format. Expected plain text."},
                            status=status.HTTP_400_BAD_REQUEST)

        if raw_data  == "get_predictions":

            # Graciously extracting inference data

            if not os.path.exists(MODEL_PATH):
                print("🚫 Model file not found.")
                return Response("Model file missing.", status=status.HTTP_500_INTERNAL_SERVER_ERROR)


            model = joblib.load(MODEL_PATH)
            inference_data = RoadAnomalyInferenceLogs.objects.order_by("id").values()
            df = pd.DataFrame(inference_data)
            if df.empty:
                return Response(f"No Inference Data, size of data => {len(raw_data)}", status=status.HTTP_400_BAD_REQUEST )

            # print("Gracious output:", df.shape)
            data_globally_aligned = align_to_global_frame(df)
            batched_df = fix_batches(data_globally_aligned)
            engineered_df = apply_feature_extraction_across_all_identical_anomaly_batches(batched_df)
            predictions = ml_pipeline(engineered_df)
            print(predictions)

            for anomaly_class, percentage in predictions.items():
                data = {
                    "batch_id":None,
                    "timestamp": str(datetime.now()),  # Or use str(datetime.now()) for uniform time
                    "log_interval": None,
                    "acc_x": None,
                    "acc_y": None,
                    "acc_z": None,
                    "rot_x": None,
                    "rot_y": None,
                    "rot_z": None,
                    "latitude": None,
                    "longitude": None,
                    "speed": None,
                    "accuracy": None,  # Keep this as you intended,
                    "anomaly_prediction" : f"{anomaly_class} ({percentage:.2f}%)"
                }

                serializer = self.get_serializer(data = data)
                serializer.is_valid(raise_exception = True)
                serializer.save()

            queryset = RoadAnomalyPredictionOutput.objects.all().order_by("id")
            full_serializer = self.get_serializer(queryset, many = True)

            # RoadAnomalyInferenceLogs.objects.all().delete()       
            return Response(full_serializer.data, status = status.HTTP_200_OK)

        else:
            return Response(f"Invalid received request: {raw_data}", status=201)

router = routers.DefaultRouter()
router.register(r"road_anomaly_predict", RoadAnomalyPredictionOutputViewSet)

urlpatterns = [
    path("",include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))

]

if __name__ == "__main__":
    inference_data = RoadAnomalyInferenceLogs.objects.order_by("id").values()
    df = pd.DataFrame(inference_data)
    # print("Gracious output:", df.shape)
    data_globally_aligned = align_to_global_frame(df)
    batched_df = fix_batches(data_globally_aligned)
    engineered_df = apply_feature_extraction_across_all_identical_anomaly_batches(batched_df)
    predictions = ml_pipeline(engineered_df)
    print(predictions)