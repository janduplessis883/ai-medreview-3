from django.db import models


class Review(models.Model):
    """Reserved for the future database-backed ingestion path."""

    review_id = models.PositiveIntegerField(unique=True)
    review_date = models.DateTimeField(null=True)
    pcn = models.CharField(max_length=255)
    surgery = models.CharField(max_length=255)
    text = models.TextField()

