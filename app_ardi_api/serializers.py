from rest_framework import serializers 


class ImageDataSerializer(serializers.Serializer): 
    img_dict = serializers.JSONField() 
