from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import subprocess
from django.templatetags.static import static

from django.shortcuts import render


from django.http import HttpResponse
from django.conf import settings

from django.shortcuts import render
import pandas as pd


import random
import networkx as nx

from geopy.distance import geodesic
from .ant_colony_class import AntColony  # Assuming you have an AntColony implementation
from django.conf import settings  
# Create your views here.
def Acceuil(request):
    return render(request,"acceuil.html")







# views.py


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







def execute_script(request):
    # Exécute le script Python
    result = subprocess.run(['python', 'app/static/wilaya.py'], capture_output=True, text=True)
    script_output = result.stdout

    return render(request, 'script_output.html', {'script_output': script_output})





#########################################wilayaa#########################################################
    


    
data = pd.read_excel('Cordonnees_GPS.xlsx')
n = len(data)
graph = np.zeros((n, n))
def wilaya():
        

    distances = {}
    for i, city1 in enumerate(data['Ville']):
        for j, city2 in enumerate(data['Ville']):
            if i != j:
                coords1 = (data.loc[i, 'Latitude'], data.loc[i, 'Longitude'])
                coords2 = (data.loc[j, 'Latitude'], data.loc[j, 'Longitude'])
                distance = geodesic(coords1, coords2).kilometers
                distances[(city1, city2)] = distance

   
    
    for i in range(n):
        for j in range(n):
            if i != j:
                graph[i][j] = distances[(data.loc[i, 'Ville'], data.loc[j, 'Ville'])]

    ville_depart = 'Nouakchott'
    ville_arrivee = 'Nouakchott'

    index_ville_depart = data[data['Ville'] == ville_depart].index[0]

    colony = AntColony(graph, n_ants=1000, max_iter=100, alpha=1, beta=2, rho=0.5, Q=100)
    best_path, best_distance = colony.run(start=index_ville_depart)

    index_ville_arrivee = best_path.index(index_ville_depart)
    chemin_noms_villes = [data.loc[indice, 'Ville'] for indice in best_path[index_ville_arrivee:]]
    chemin_noms_villes += [data.loc[indice, 'Ville'] for indice in best_path[:index_ville_arrivee]]
    chemin_noms_villes.append(ville_depart)  # Ajouter Nouakchott à la fin du chemin

    chemin_separe = " -> ".join(chemin_noms_villes)

    # print("Meilleur chemin trouvé par l'ACO de ", ville_depart, "à", ville_arrivee, ":", chemin_separe)
    # print("Distance du meilleur chemin de ", ville_depart, "à", ville_arrivee, ":", best_distance)


############################################### Mougataa #########################
    

    


    
def mougataa():



    def calculate_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers



    excel_path ='Cordonnees_GPS.xlsx'

    data_mougataa = pd.read_excel(excel_path, sheet_name='Mougataa')

    def get_city_info(city_name):
        return data_mougataa[data_mougataa['nom'] == city_name].iloc[0]

    ville_depart = "Arafat"

    depart_info = get_city_info(ville_depart)

    print(f"Mougataa de départ: {ville_depart} ({depart_info['wilaya']})")

    villes_a_parcourir = [ville_depart]

    wilayas = data_mougataa['wilaya'].unique()
    for wilaya in wilayas:
        villes_wilaya = data_mougataa[data_mougataa['wilaya'] == wilaya]['nom'].tolist()
        random.shuffle(villes_wilaya)  # Mélanger l'ordre des villes dans chaque wilaya
        for ville in villes_wilaya:
            if ville not in villes_a_parcourir:
                villes_a_parcourir.append(ville)
    villes_a_parcourir.append(ville_depart)

    chemin_parcouru = ' -> '.join(villes_a_parcourir)
    print(f"Chemin parcouru : {chemin_parcouru}")

    distance_totale = 0
    for i in range(len(villes_a_parcourir) - 1):
        ville_depart_info = get_city_info(villes_a_parcourir[i])
        ville_arrivee_info = get_city_info(villes_a_parcourir[i + 1])
        distance_totale += calculate_distance(ville_depart_info['Latitude'], ville_depart_info['Longitude'],
                                            ville_arrivee_info['Latitude'], ville_arrivee_info['Longitude'])

    print(f"Distance totale du chemin parcouru : {distance_totale} kilomètres")
    ## Créer un graphe représentant les distances entre les mougataas
    n_mougataas = len(data_mougataa)
    distances = np.zeros((n_mougataas, n_mougataas))

    # Calculer les distances entre chaque paire de mougataas
    for i in range(n_mougataas):
        for j in range(i + 1, n_mougataas):
            mougataa1 = data_mougataa.iloc[i]
            mougataa2 = data_mougataa.iloc[j]
            distance = calculate_distance(mougataa1['Latitude'], mougataa1['Longitude'],
                                        mougataa2['Latitude'], mougataa2['Longitude'])
            distances[i, j] = distances[j, i] = distance

    # Initialiser l'algorithme ACO
    colony = AntColony(graph=distances, n_ants=10, max_iter=100, alpha=1, beta=2, rho=0.5, Q=100)

    # Exécuter l'algorithme ACO à partir de la mougataa de départ
    best_path, best_distance = colony.run()

    print("Chemin optimal trouvé par l'algorithme ACO :")
    chemin_optimal = " -> ".join(data_mougataa.iloc[idx]['nom'] for idx in best_path)
    print(chemin_optimal)

    print(f"Distance totale du chemin optimal : , {best_distance} kilomètres")


# ####################################le graphe##############################################
# # # Création d'un graphique NetworkX à partir du graphe des villes
    





G = nx.Graph()

# Ajout des nœuds (villes)
for i, ville in enumerate(data['Ville']):
    G.add_node(i, label=ville)

    # Ajout des arêtes (connexions entre les villes)
for i in range(n):
    for j in range(n):
        if i != j:
             G.add_edge(i, j, weight=graph[i][j])

    # Création d'un dictionnaire de positions pour afficher les nœuds sur le graphique
    positions = {i: (data.loc[i, 'Longitude'], data.loc[i, 'Latitude']) for i in range(n)}



# Affichage du graphique avec les noms des villes comme étiquettes
def affichage_Graph(request):
    mougataa()
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=300, font_size=8)
    plt.title("Graphique des connexions entre les villes")
    
    static_path = os.path.join(settings.STATIC_ROOT, 'graphs')
    if not os.path.exists(static_path):
        os.makedirs(static_path)
    graph_image_path = os.path.join(static_path, 'graph_image.png')
    plt.savefig(graph_image_path)

    plt.close()

    # Create Plotly graph from image file
    plotly_graph = go.Figure(go.Image(source=graph_image_path))
    plotly_div = plotly_graph.to_html(full_html=False)

    return render(request, "graph.html", {"plotly_div": plotly_div})



##esma

# import pandas as pd
# import networkx as nx
import geopy.distance
# import matplotlib.pyplot as plt
# Charger les coordonnées GPS des moughataa depuis le fichier Excel


def charger_coordonnees_excel(fichier_excel):
    data = pd.read_excel(fichier_excel,"Mougataa")
    return data.set_index('nom').to_dict(orient='index')
def calculer_distance(moughataa1, moughataa2, coords):
    coords_moughataa1 = (coords[moughataa1]['Latitude'], coords[moughataa1]['Longitude'])
    coords_moughataa2 = (coords[moughataa2]['Latitude'], coords[moughataa2]['Longitude'])
    return geopy.distance.geodesic(coords_moughataa1, coords_moughataa2).km

# Construire le graphe non-orienté pondéré avec les moughataa comme nœuds
def construire_graphe(coords):
    G = nx.Graph()
    for moughataa1 in coords:
        for moughataa2 in coords:
            if moughataa1 != moughataa2:
                distance = calculer_distance(moughataa1, moughataa2, coords)
                G.add_edge(moughataa1, moughataa2, weight=distance)
    return G

# Algorithme d'approximation pour le TSP avec une moughataa comme ville de départ
def tsp_approximation(moughataa_depart, coords):
    G = construire_graphe(coords)
    mst = nx.minimum_spanning_tree(G)
    dfs_edges = list(nx.dfs_edges(mst, source=moughataa_depart))
    chemin = [moughataa_depart]
    distance_totale = 0
    moughataas_visitees = {moughataa_depart}
    wilaya_actuelle = coords[moughataa_depart]['wilaya']
    moughataas_wilaya = [moughataa_depart] + [moughataa for moughataa, coord in coords.items() if coord['wilaya'] == wilaya_actuelle and moughataa != moughataa_depart]
    for moughataa in moughataas_wilaya:
        if moughataa != moughataa_depart:
            distance_totale += G[chemin[-1]][moughataa]['weight']
            chemin.append(moughataa)
            moughataas_visitees.add(moughataa)
    for edge in dfs_edges:
        next_moughataa = edge[1]
        wilaya_next_moughataa = coords[next_moughataa]['wilaya']
        if wilaya_next_moughataa not in wilaya_actuelle:  # Correction ici
            moughataas_wilaya = [moughataa for moughataa, coord in coords.items() if coord['wilaya'] == wilaya_next_moughataa and moughataa not in moughataas_visitees]
            for moughataa in moughataas_wilaya:
                chemin.append(moughataa)
                moughataas_visitees.add(moughataa)
                distance_totale += G[edge[0]][edge[1]]['weight']
    distance_totale += G[chemin[-1]][moughataa_depart]['weight']  # Ajouter la distance pour retourner à la moughataa de départ
    chemin.append(moughataa_depart)
    return chemin, distance_totale





# Fonction pour afficher le graphe des rues avec les chemins de tournée pour les moughataa







# Charger les coordonnées GPS des moughataa depuis le fichier Excel
fichier_excel = "Cordonnees_GPS.xlsx"
coords = charger_coordonnees_excel(fichier_excel)

def saisir_moughataa(coords):
    moughataa = "Arafat"
    if moughataa in coords.keys() and coords[moughataa]['wilaya'] in ['Nouakchott-Nord', 'Nouakchott-Sud', 'Nouakchott-Ouest']:
        return moughataa
    else:
        print("Le moughataa saisi n'appartient pas aux wilayas autorisées (Nouakchott-Nord, Nouakchott-Sud, Nouakchott-Ouest).")
        return saisir_moughataa(coords)

moughataa_depart = saisir_moughataa(coords)
# # Résoudre le problème du voyageur de commerce avec l'algorithme d'approximation
chemin, distance_totale = tsp_approximation(moughataa_depart, coords)

# # Afficher le chemin de tournée approximatif
# print("Chemin de tournée approximatif en partant et en terminant à", moughataa_depart, ":")
# print(" -> ".join(chemin))
# print("Distance totale parcourue:", distance_totale, "km")

# # Afficher le graphe des rues avec les chemins de tournée pour les moughataa
# afficher_graphe_chemin(coords, chemin, moughataa_depart)

def afficher_graphe_chemin(request):
    fichier_excel = "Cordonnees_GPS.xlsx"
    coords = charger_coordonnees_excel(fichier_excel)
    
    moughataa_depart = "Arafat"  # Or get it dynamically from the request
    
    chemin, distance_totale = tsp_approximation(moughataa_depart, coords)
    
    chemin_coords = [coords[moughataa] for moughataa in chemin]
    lats = [coord['Latitude'] for coord in chemin_coords]
    lons = [coord['Longitude'] for coord in chemin_coords]

    # Create a Plotly scatter mapbox figure
    fig = go.Figure(go.Scattermapbox(
        mode='markers+lines',
        lat=lats,
        lon=lons,
        marker={'size': 10},
        line=dict(width=2, color='red'),
    ))

    # Add the starting point as a green marker
    fig.add_trace(go.Scattermapbox(
        mode='markers',
        lat=[coords[moughataa_depart]['Latitude']],
        lon=[coords[moughataa_depart]['Longitude']],
        marker={'size': 12, 'color': 'green'},
        name='Moughataa de départ',
    ))

    # Define layout for the map
    fig.update_layout(
        title="Chemin de tournée sur la carte",
        mapbox=dict(
            style="open-street-map",
            zoom=10,
            center=dict(lat=lats[0], lon=lons[0]),
        ),
        showlegend=True,
    )

    # Convert the Plotly figure to HTML
    plot_div = fig.to_html(full_html=False, include_plotlyjs=False)

    context = {'plot_div': plot_div}
    return render(request, 'graph.html', context)




#esmaWilya


def charger_coordonnees_excel(fichier_excel):
    data = pd.read_excel(fichier_excel)
    return data.set_index('Ville').to_dict(orient='index')

# Calculer la distance entre deux villes en utilisant les coordonnées GPS
def calculer_distance(ville1, ville2, coords):
    coords_ville1 = (coords[ville1]['Latitude'], coords[ville1]['Longitude'])
    coords_ville2 = (coords[ville2]['Latitude'], coords[ville2]['Longitude'])
    return geopy.distance.geodesic(coords_ville1, coords_ville2).km

# Construire le graphe non-orienté pondéré
def construire_graphe(coords):
    G = nx.Graph()
    for ville1 in coords:
        for ville2 in coords:
            if ville1 != ville2:
                distance = calculer_distance(ville1, ville2, coords)
                G.add_edge(ville1, ville2, weight=distance)
    return G

# Algorithme d'approximation pour le TSP
def tsp_approximation(ville_depart, coords):
    G = construire_graphe(coords)
    mst = nx.minimum_spanning_tree(G)
    dfs_edges = list(nx.dfs_edges(mst, source=ville_depart))
    chemin = [ville_depart]
    distance_totale = 0
    villes_visitees = {ville_depart}
    for edge in dfs_edges:
        next_ville = edge[1]
        if next_ville not in villes_visitees:
            chemin.append(next_ville)
            villes_visitees.add(next_ville)
            distance_totale += G[edge[0]][edge[1]]['weight']
    distance_totale += G[chemin[-1]][ville_depart]['weight']  # Ajouter la distance pour retourner à la ville de départ
    chemin.append(ville_depart)
    return chemin, distance_totale

# Entrée du nom de la ville de départ
ville_depart = "Nouakchott"

# Charger les coordonnées GPS depuis le fichier Excel
fichier_excel = "Cordonnees_GPS.xlsx"
coords = charger_coordonnees_excel(fichier_excel)

# Résoudre le problème du voyageur de commerce avec l'algorithme d'approximation
chemin, distance_totale = tsp_approximation(ville_depart, coords)

# Fonction pour afficher le graphe des rues avec les chemins de tournée
def afficher_graphe_chemin(coords, chemin, ville_depart):
    # Extraire les coordonnées des villes dans l'ordre du chemin de tournée
    chemin_coords = [coords[ville] for ville in chemin]

    # Extraire les coordonnées en latitude et longitude
    lats = [coord['Latitude'] for coord in chemin_coords]
    lons = [coord['Longitude'] for coord in chemin_coords]

    # Créer la figure Plotly
    fig = go.Figure()

    # Ajouter les points correspondant aux villes
    fig.add_trace(go.Scattermapbox(
        mode='markers+lines',
        lat=lats,
        lon=lons,
        marker={'size': 10},
        line=dict(width=2, color='red'),
    ))

    # Ajouter la ville de départ avec un symbole différent
    fig.add_trace(go.Scattermapbox(
        mode='markers',
        lat=[coords[ville_depart]['Latitude']],
        lon=[coords[ville_depart]['Longitude']],
        marker={'size': 12, 'color': 'green'},
        name='Ville de départ',
    ))

    # Mettre en forme la carte
    fig.update_layout(
        title="Chemin de tournée sur la carte",
        mapbox=dict(
            style="open-street-map",
            zoom=10,
            center=dict(lat=lats[0], lon=lons[0]),
        ),
        showlegend=True,
    )

    # Convertir la figure Plotly en HTML
    plot_div = fig.to_html(full_html=False)

    # Rendre la page HTML avec le graphique Plotly
    context = {'plot_div': plot_div}
    return plot_div

# Afficher le graphe des rues avec les chemins de tournée

def afficher_grph2(request):
    plot_div=afficher_graphe_chemin(coords, chemin, ville_depart)

    context = {'plot_div': plot_div}
    return render(request, 'graph.html', context)



    