from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.Acceuil,name="acceuil"),
    path('visualiser-excel/', views.visualiser_excel, name='visualiser_excel'),
    path('upload/', views.upload_file, name='upload_file'),
    path('telecharger-excel/', views.telecharger_excel, name='telecharger_excel'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
