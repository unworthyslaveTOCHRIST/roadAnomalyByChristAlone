#ALL THANKS AND GLORY TO THE AND my ONLY GOD AND LORD JESUS CHRIST ALONE

from django.db import models

# Create your models here.

class RoadAnomalyInput(models.Model):

    ANOMALY_CHOICES = [
        ('smooth', 'Smooth Segment'),
        ('crack', 'Crack Segment'),
        ('bump', 'Bump Segment'),
        ('road-patch', 'Road Patch Segment'),
        ('pothole_mild', 'Mild Pothole Segment'),
        ('pothole_severe', 'Severe Pothole Segment'),


    ]

    anomaly = models.CharField(
        max_length=20,
        choices=ANOMALY_CHOICES,
        default='no_defect',
        help_text="Classify the road anomaly based on severity."
    )

    latitude = models.FloatField()
    longitude = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Road Anomaly @{self.timestamp}  | Anomaly Type : {self.anomaly}"