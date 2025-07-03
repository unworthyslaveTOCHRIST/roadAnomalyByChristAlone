# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

from django.db import models

# Create your models here.


# Create your models here.
class RoadAnomalyPredictionOutput(models.Model):
    batch_id = models.BigIntegerField(null=True, blank=True)
    acc_x = models.FloatField(null=True, blank=True)
    acc_y = models.FloatField(null=True, blank=True)
    acc_z = models.FloatField(null=True, blank=True)
    rot_x = models.FloatField(null=True, blank=True)
    rot_y = models.FloatField(null=True, blank=True)
    rot_z = models.FloatField(null=True, blank=True)
    speed = models.FloatField(null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    timestamp = models.TextField()
    log_interval = models.BigIntegerField(null=True, blank=True)
    anomaly_prediction = models.TextField()


    def __str__(self):
        return f"Road Anomaly Prediction Data received from Django Ml Model @{self.timestamp}  | anomaly : {self.anomaly_prediction}"
