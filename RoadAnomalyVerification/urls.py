# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from django.urls import path, include
from rest_framework import serializers, viewsets, routers
from .models import RoadAnomalyVerification
from rest_framework.response import Response
from rest_framework import status
from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs
from RoadAnomalyPredictionOutput.models import RoadAnomalyPredictionOutput
from rest_framework.parsers import JSONParser
from .parsers import PlainTextParser


class RoadAnomalyVerificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoadAnomalyVerification
        fields = ("id","batch_id","acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "speed", "timestamp", "log_interval","latitude","longitude","accuracy","anomaly_prediction", "response")


class RoadAnomalyVerificationViewSet(viewsets.ModelViewSet):
    queryset = RoadAnomalyVerification.objects.all()
    serializer_class = RoadAnomalyVerificationSerializer
    parser_classes = [PlainTextParser, JSONParser]  # <--- Graciously accepts both JSON and plain text --->
    
    
    def create(self, request, *args, **kwargs):
        raw_data = request.data
        if not isinstance(raw_data,str):
            return Response({"error": "Invalid data format. Expected plain text."},
                            status=status.HTTP_400_BAD_REQUEST)


        if raw_data == "accept":

            # To Graciously later include inference data into RoadAnomalyInput grouping identical instances first
            # Then Graciously delete all inference data and associated predictions
            return Response(f"Verification message received:{raw_data}", status = status.HTTP_200_OK)  
            
            

        elif raw_data == "reject":

            # To only Graciously delete all inference data and associated predictions
            return Response(f"Verification message received:{raw_data}", status = status.HTTP_200_OK)  
        

        # serializer = self.get_serializer(data=request.data, many=True)
        # serializer.is_valid(raise_exception=True)
        # self.perform_create(serializer)
        # return Response(serializer.data, status=201)


router = routers.DefaultRouter()
router.register(r"road_anomaly_verify", RoadAnomalyVerificationViewSet)

urlpatterns = [
    path("",include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))

]