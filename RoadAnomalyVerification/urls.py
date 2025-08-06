# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE
from django.urls import path, include
from rest_framework import serializers, viewsets, routers
from .models import RoadAnomalyVerification
from rest_framework.response import Response
from rest_framework import status

from RoadAnomalyInferenceLogs.models import RoadAnomalyInferenceLogs
from RoadAnomalyPredictionOutput.models import RoadAnomalyPredictionOutput
from RoadAnomalyInput.models import RoadAnomalyInput

from rest_framework.parsers import JSONParser
from .parsers import PlainTextParser
import pandas as pd
import numpy as np
from datetime import datetime

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
            # predictions = pd.read_csv("predictions.csv")
            # queryset_count_former = RoadAnomalyInput.objects.all().order_by("id").count
            # df = pd.DataFrame(predictions)
            # if df.empty:
            #     return Response("No Predictions available", status=status.HTTP_400_BAD_REQUEST )

            # for i in range(df.shape[0]):
            #     row = df.iloc[i]   #Graciously getting each row of prediction information

            #     anomaly     =       f"{row["predictions"]} {row["confidence_in_%"]}%"
            #     latitude    =       row["latitude"]
            #     longitude   =       row["longitude"]
            #     timestamp   =       str(datetime.now())

            #     # Graciously defining lookup fields based on which duplicated-location entries are prevented
            #     lookup_fields = {
            #         "latitude"  :   latitude,
            #         "longitude" :   longitude
            #     }

            #     # Graciously defining the remaining fields to update or insert
            #     defaults = {
            #         "timestamp": timestamp,  # Or use str(datetime.now()) for uniform time              
            #         "anomaly" : anomaly,
            #         #To graciously include distance covered during period of observation per inference data-batch
            #     }
                
            #     RoadAnomalyInput.objects.update_or_create(
            #         **lookup_fields,
            #         defaults=defaults
            #     )

            # queryset = RoadAnomalyInput.objects.all().order_by("id")
            # queryset_count_new = RoadAnomalyInput.objects.all().order_by("id").count
            # full_serializer = self.get_serializer(queryset, many = True)

            # # RoadAnomalyInferenceLogs.objects.all().delete()       
            # # return Response(full_serializer.data, status = status.HTTP_200_OK)
            # # To Graciously later include inference data into RoadAnomalyInput grouping identical instances first
            # # Then Graciously delete all inference data and associated predictions
            # return Response(f"Verification message received, no of anomalies:{queryset.count}\n No of Anomalies(prev):{queryset_count_former}, No of Anomalies(current): {queryset_count_new}", status = status.HTTP_200_OK)  
            return Response(f"Vefification message graciously received",status=status.HTTP_200_OK)
            

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