# ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE


from django.db import models

# Create your models here.
class RoadAnomalyManualDataCollection(models.Model):
    batch_id = models.BigIntegerField()
    acc_x = models.FloatField()
    acc_y = models.FloatField()
    acc_z = models.FloatField()
    rot_x = models.FloatField()
    rot_y = models.FloatField()
    rot_z = models.FloatField()
    speed = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    accuracy = models.FloatField()
    timestamp = models.TextField()
    log_interval = models.BigIntegerField()
    anomaly = models.TextField()

    def __str__(self):
        return f"Road Anomaly Manually Collected Data received from App @{self.timestamp}  | speed : {self.speed}"
