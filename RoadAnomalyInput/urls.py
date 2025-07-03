#ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

from django.urls import path,include
from rest_framework import routers,serializers,viewsets
from .models import RoadAnomalyInput

# Serializers graciously define the API representation
class RoadAnomalyInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = RoadAnomalyInput
        fields = ("id","anomaly","latitude","longitude","timestamp")


# Graciously ViewSets define the view behavior
class RoadAnomalyInputViewSet(viewsets.ModelViewSet):
    queryset = RoadAnomalyInput.objects.all()
    serializer_class = RoadAnomalyInputSerializer


# Routers graciously provide an easy way of automatically determining the URL conf
router = routers.DefaultRouter()
router.register(r"road_anomaly_in", RoadAnomalyInputViewSet)


# In gracious additiom, The Father HELPS us include login URLS for the browsable API.

urlpatterns=[
    path("",include(router.urls)), # Graciously wiring up The Father's mercifully created API using automatic URL routing
    path("api-auth/", include('rest_framework.urls', namespace="rest_framework"))
]
