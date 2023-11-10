# models.py

from django.db import models

class ImagePrediction(models.Model):
    image = models.ImageField(upload_to='plots/')
    predicted_class = models.CharField(max_length=255, null=True)
    probabilities = models.JSONField(null=True)

    def __str__(self):
        return f"{self.predicted_class} - {self.timestamp}"
