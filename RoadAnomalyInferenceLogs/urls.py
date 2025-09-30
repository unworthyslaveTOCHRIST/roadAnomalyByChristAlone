# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from django.urls import path, include
from rest_framework import serializers, viewsets, routers
from .models import RoadAnomalyInferenceLogs
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
from .parsers import PlainTextParser

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# Graciously setting up Django environment around script

MODEL_PATH = 'road_anomaly.pkl'


class RoadAnomalyInferenceLogsSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoadAnomalyInferenceLogs
        fields = ("id","batch_id","acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "speed", "timestamp", "log_interval","latitude","longitude","accuracy")


class RoadAnomalyInferenceLogsViewSet(viewsets.ModelViewSet):
    queryset = RoadAnomalyInferenceLogs.objects.all()
    serializer_class = RoadAnomalyInferenceLogsSerializer
    parser_classes = [PlainTextParser, JSONParser]  # <--- Graciously accepts both JSON and plain text --->


    def create(self, request, *args, **kwargs):
        raw_data = request.data

        if raw_data == "clean_up_inference_database":
            RoadAnomalyInferenceLogs.objects.all().delete()
            return Response(f" Inference database emptied ", status=status.HTTP_201_CREATED)

        if not isinstance(raw_data, str):
            return Response({"error": "Invalid data format. Expected plain text."},
                            status=status.HTTP_400_BAD_REQUEST)

        lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
        line_count = len(lines)
        saved_items = []

        for i, line in enumerate(lines):
            print(f"📥 Line {i + 1}/{line_count}: {line}")
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 13:
                    print(f"❌ Skipping line {i + 1} (only {len(parts)} parts): {line}")
                    continue

                data = {
                    "batch_id": int(parts[0]),
                    "timestamp": str(datetime.now()),  # Or use str(datetime.now()) for uniform time
                    "log_interval": int(parts[12]),
                    "acc_x": float(parts[1]),
                    "acc_y": float(parts[2]),
                    "acc_z": float(parts[3]),
                    "rot_x": float(parts[4]),
                    "rot_y": float(parts[5]),
                    "rot_z": float(parts[6]),
                    "latitude": float(parts[8]),
                    "longitude": float(parts[9]),
                    "speed": float(parts[7]),
                    "accuracy": float(parts[10]),  # Keep this as you intended
                }

                serializer = self.get_serializer(data=data)
                serializer.is_valid(raise_exception=True)
                serializer.save()
                saved_items.append(serializer.data)

            except Exception as e:
                print(f"⚠️ Error in line {i + 1}: {line}\n  ↳ Exception: {e}")
                continue

        print(f"✅ Successfully saved {len(saved_items)} of {line_count} lines.")
        return Response(f"Received {len(saved_items)} more rows, Current Inference Size : {RoadAnomalyInferenceLogs.objects.all().count()} rows", status=status.HTTP_201_CREATED)
        # return Response(f"{len(parts)} fields", status=status.HTTP_201_CREATED)



router = routers.DefaultRouter()
router.register(r"road_anomaly_infer", RoadAnomalyInferenceLogsViewSet)

urlpatterns = [
    path("",include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))

]

if __name__ == "__main__":
    RoadAnomalyInferenceLogsViewSet.run_predictions()