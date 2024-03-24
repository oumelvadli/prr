from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.Acceuil,name="acceuil"),
    path('visualiser-excel/', views.visualiser_excel, name='visualiser_excel'),
    path('upload/', views.upload_file, name='upload_file'),
    path('telecharger-excel/', views.telecharger_excel, name='telecharger_excel'),
    path('afficher-graph/', views.affichage_Graph, name='affichier'),
    path('afficher-graph-esma/', views.afficher_graphe_chemin, name='affichier_graph'),
    path('afficher-graph-2/', views.afficher_grph2, name='affichier_graph2'),

]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
