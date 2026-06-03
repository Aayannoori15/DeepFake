from django.urls import path

from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('image/', views.image_detector, name='image_detector'),
    path('text/', views.text_detector, name='text_detector'),
    path('plagiarism/', views.Plagirism_report, name='Plagirism_report'),
]
