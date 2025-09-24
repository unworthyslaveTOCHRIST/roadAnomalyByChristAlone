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


        if raw_data == "accept" or raw_data == "accept_without_data_wipe":
            predictions = pd.read_csv("predictions.csv")
            queryset_count_former = RoadAnomalyInput.objects.all().order_by("id").count()
            df = pd.DataFrame(predictions)
            if df.empty:
                return Response("No Predictions available", status=status.HTTP_400_BAD_REQUEST )

            for i in range(df.shape[0]):
                row = df.iloc[i]   #Graciously getting each row of prediction information

                anomaly                 =       f"{row['predictions']}"
                latitude                =       row["latitude"]
                longitude               =       row["longitude"]
                timestamp               =       row["inference_start_time"]

                # Graciously defining lookup fields based on which duplicated-location entries are prevented
                lookup_fields = {
                    "latitude"  :   latitude,
                    "longitude" :   longitude
                }

                # Graciously defining the remaining fields to update or insert
                defaults = {
                    "timestamp": timestamp,  # Or use str(datetime.now()) for uniform time              
                    "anomaly" : anomaly,
                    #To graciously include distance covered during period of observation per inference data-batch
                }
                
                RoadAnomalyInput.objects.update_or_create(
                    **lookup_fields,
                    defaults=defaults
                )

            queryset = RoadAnomalyInput.objects.all().order_by("id")
            queryset_count_new = RoadAnomalyInput.objects.all().order_by("id").count()
            # full_serializer = self.get_serializer(queryset, many = True)
            if raw_data == "accept_without_data_wipe":
                pass
            elif raw_data == "accept":
                RoadAnomalyInferenceLogs.objects.all().delete() 
            
            inference_data_count = RoadAnomalyInferenceLogs.objects.all().count() if RoadAnomalyInferenceLogs.objects.all().count()  >  0 else "No"
            RoadAnomalyPredictionOutput.objects.all().delete() 
            predictions_count = RoadAnomalyPredictionOutput.objects.all().count() if RoadAnomalyPredictionOutput.objects.all().count()  >  0 else "No"     

            return Response(f"Verification message received, No of Anomalies(prev):{queryset_count_former}, No of Anomalies(current): {queryset_count_new}, Inference data : {inference_data_count} logs, Predictions: {predictions_count}", 
                            status = status.HTTP_200_OK)  
            
            

        elif raw_data == "reject" or raw_data == "reject_without_data_wipe":

            # To Graciously 
            # delete all inference data and associated predictions

            if raw_data == "reject_without_data_wipe":
                pass
            elif raw_data == "reject":
                RoadAnomalyInferenceLogs.objects.all().delete()     
            inference_data_count = RoadAnomalyInferenceLogs.objects.all().count() if RoadAnomalyInferenceLogs.objects.all().count()  >  0 else "None"
            RoadAnomalyPredictionOutput.objects.all().delete() 

            predictions_count = RoadAnomalyPredictionOutput.objects.all().count() if RoadAnomalyPredictionOutput.objects.all().count()  >  0 else "None"     

            return Response(f"Verification message received, Inference data : {inference_data_count} logs, Predictions: {predictions_count}", 
                            status = status.HTTP_200_OK)
        



router = routers.DefaultRouter()
router.register(r"road_anomaly_verify", RoadAnomalyVerificationViewSet)

urlpatterns = [
    path("",include(router.urls)),
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))

]