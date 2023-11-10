from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='classify_and_save_image'),
    path('clear/', views.clear_uploads, name='clear_uploads'),
    path('prediction/', views.prediction_results, name='prediction_results'),  # Add this line
  # Add this line
]
