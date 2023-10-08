from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('model', views.model, name="model"),
    path('add_data', views.add_data, name="add_data")
]