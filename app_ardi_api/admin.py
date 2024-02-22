from django.contrib import admin
from . models import ImageData

# Register your models here.
@admin.register(ImageData) 
class ImageDataAdmin(admin.ModelAdmin): 
    list_display = ['img_dict'] 