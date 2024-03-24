from django import forms

class UploadForm(forms.Form):
    fichier_excel = forms.FileField(label='Sélectionner un fichier Excel')
