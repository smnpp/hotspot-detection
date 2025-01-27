import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import colorsys
from collections import Counter
from scipy.spatial import ConvexHull
import webbrowser
import os

def main():
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('datasetCleaned.csv', low_memory=False)
    df = df.head(10000)  # Limiter à 10000 points pour commencer
    print(f"Taille du dataset: {len(df)}")
    
    # Préparer les données pour le clustering
    X = df[['lat', 'long']].values
    
    # Paramètres du clustering DBSCAN
    eps = 0.0004  # Rayon de voisinage
    min_samples = 5  # Nombre minimum de points pour former un cluster
    
    # Appliquer DBSCAN
    print("Application de DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(X)
    
    # Nombre de clusters trouvés (excluant le bruit qui est -1)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"Nombre de clusters trouvés : {n_clusters}")
    
    # Générer les couleurs pour les clusters
    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(color)
    colors = ['#808080'] + colors  # Ajouter une couleur grise pour les points de bruit
    
    # Créer la carte
    print("Création de la carte...")
    carte = folium.Map(location=[df['lat'].mean(), df['long'].mean()], zoom_start=12)
    
    # Créer les zones de clusters
    for cluster_id in range(n_clusters):
        mask = df['cluster'] == cluster_id
        cluster_points = X[mask]
        
        # Vérifier qu'il y a assez de points pour former un polygone
        unique_points = np.unique(cluster_points, axis=0)
        if len(unique_points) >= 3:
            try:
                jittered_points = unique_points + np.random.normal(0, 1e-10, unique_points.shape)
                hull = ConvexHull(jittered_points)
                hull_points = jittered_points[hull.vertices]
                polygon_points = [[point[0], point[1]] for point in hull_points]
                
                folium.Polygon(
                    locations=polygon_points,
                    color=colors[cluster_id + 1],
                    weight=2,
                    fill=True,
                    fill_color=colors[cluster_id + 1],
                    fill_opacity=0.2,
                    popup=f'Cluster {cluster_id}'
                ).add_to(carte)
            except Exception as e:
                print(f"Impossible de créer le polygone pour le cluster {cluster_id}: {e}")
    
    # Ajouter les points
    for idx, row in df.iterrows():
        color_idx = row['cluster'] + 1 if row['cluster'] >= 0 else 0
        cluster_name = f"Cluster {row['cluster']}" if row['cluster'] >= 0 else "Non clustérisé"
        
        # Créer le lien Flickr
        flickr_link = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
        
        # Créer le popup
        popup_content = f"""
        <div style="min-width: 200px;">
        Cluster: {cluster_name}<br>
        <a href="{flickr_link}" target="_blank">Voir la photo sur Flickr</a>
        </div>
        """
        
        # Ajuster la taille et l'opacité selon que le point est dans un cluster ou non
        radius = 3 if row['cluster'] >= 0 else 3
        opacity = 0.7 if row['cluster'] >= 0 else 0.15
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=radius,
            color=colors[color_idx],
            fill=True,
            popup=popup_content,
            fill_opacity=opacity
        ).add_to(carte)
    
    # Sauvegarder et ouvrir la carte
    print("Sauvegarde de la carte...")
    carte.save('carte_clusters_dbscan.html')
    webbrowser.open('file://' + os.path.realpath('carte_clusters_dbscan.html'))

if __name__ == "__main__":
    main()