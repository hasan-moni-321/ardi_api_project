from django.db import models
import json 

# Create your models here.
class ImageData(models.Model): 
    img_dict = models.JSONField() 
