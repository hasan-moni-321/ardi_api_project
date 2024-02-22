from django.urls import path 
from app_ardi_api import views 



urlpatterns = [
    path('', views.input_image, name='input_image'),    #image/
    #path('data/', views.getting_json_data, name='getting_json_data') 
]