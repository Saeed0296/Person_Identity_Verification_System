from django.urls import path
from . import views

app_name = 'facematchapp'

urlpatterns = [

    path('', views.home, name='home'),
    path('confusion_matrix/', views.faces_match_view, name='confusion_matrix'),
    path('face_comparision/', views.face_match_view, name='upload'),
]