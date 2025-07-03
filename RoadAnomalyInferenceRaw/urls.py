# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from django.urls import path, include
from rest_framework import serializers, viewsets, routers
from .models import RoadAnomalyInferenceRaw
from rest_framework.parsers import JSONParser
from .parsers import PlainTextParser
from rest_framework import status
from rest_framework.response import Response
from datetime import datetime



class RoadAnomalyInferenceRawSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoadAnomalyInferenceRaw
        fields = ("id","data", "timestamp")


class RoadAnomalyInferenceRawViewSet(viewsets.ModelViewSet):
    queryset = RoadAnomalyInferenceRaw.objects.all()
    serializer_class = RoadAnomalyInferenceRawSerializer
    parser_classes = [PlainTextParser, JSONParser]  # <--- Graciously accepts both JSON and plain text --->


    def create(self, request, *args, **kwargs):
        # if isinstance(request.data, str):
        try:
            raw_data = request.data
            if not isinstance(raw_data, str):
                return Response({"error" : "Invalid data format. Expected plain text."},
                            status = status.HTTP_400_BAD_REQUEST)

            # data = {
            #     "data" : raw_data,
            # }

            serializer = self.get_serializer(data=raw_data, many=True)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
                print(f"⚠️ Error: {e} \n")


router = routers.DefaultRouter()
router.register(r"road_anomaly_inference_raw", RoadAnomalyInferenceRawViewSet)

urlpatterns = [
    path("",include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))

]