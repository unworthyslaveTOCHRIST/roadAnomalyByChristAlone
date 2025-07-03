# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

from django.db import models

# Create your models here.

class RoadAnomalyInferenceRaw(models.Model):
    data = models.TextField(null = True, blank = True)

    timestamp = models.DateTimeField(auto_now = True)
    def __str__(self):
        return f"Road Anomaly Data received from App @{self.timestamp}"