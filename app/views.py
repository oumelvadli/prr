from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
from django.shortcuts import render
from .forms import UploadForm
from django.http import FileResponse
import os
from django.templatetags.static import static
from django.shortcuts import render
import pandas as pd

# Create your views here.
def Acceuil(request):
    return render(request,"acceuil.html")

def visualiser_excel(request):
    # Chemin du fichier Excel
    chemin_fichier_excel = "app/static/Cordonnees_GPS.xlsx"

    # Lire toutes les feuilles du fichier Excel
    xls = pd.ExcelFile(chemin_fichier_excel)
    feuilles = xls.sheet_names

    # Vérifier si une feuille est sélectionnée par l'utilisateur
    feuille_selectionnee = request.GET.get('feuille')

    if feuille_selectionnee and feuille_selectionnee in feuilles:
        # Lire la feuille sélectionnée
        data = pd.read_excel(chemin_fichier_excel, sheet_name=feuille_selectionnee)
    else:
        # Lire la première feuille par défaut
        data = pd.read_excel(chemin_fichier_excel)

    # Convertir les données en HTML pour affichage dans le template
    html_data = data.to_html()

    # Rendre le template avec les données HTML
    return render(request, 'visualisation_excel.html', {'html_data': html_data, 'feuilles': feuilles})



def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        # Créez un répertoire temporaire pour enregistrer le fichier
        temp_dir = os.path.join(settings.BASE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        # Enregistrez le fichier dans le répertoire temporaire
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        # Renvoyez le chemin du fichier et son URL
        file_name = uploaded_file.name
        file_url = static(file_name)
        return render(request, 'upload_success.html', {'file_name': file_name, 'file_url': file_url})
    return render(request, 'upload_form.html')

# Dans views.py

from django.http import HttpResponse
from django.conf import settings
import os

def telecharger_excel(request):
    # Chemin vers le fichier Excel à télécharger
    excel_file_path = 'app/static/Cordonnees_GPS.xlsx'

    # Vérifier si le fichier existe
    if os.path.exists(excel_file_path):
        with open(excel_file_path, 'rb') as excel_file:
            response = HttpResponse(excel_file.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="Cordonnees_GPS.xlsx"'
            return response
    else:
        return HttpResponse("Le fichier spécifié n'existe pas.")





import subprocess
from django.shortcuts import render

def execute_script(request):
    # Exécute le script Python
    result = subprocess.run(['python', 'app/static/wilaya.py'], capture_output=True, text=True)
    script_output = result.stdout

    return render(request, 'script_output.html', {'script_output': script_output})
