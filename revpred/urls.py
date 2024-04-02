from django.urls import path
from . import views

urlpatterns = [
    path('', views.web, name='web_option'),
    path('run', views.run, name='run_option'),
    path('get_history_data', views.get_history_data, name='get_history_data'),
]
