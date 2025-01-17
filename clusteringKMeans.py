import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import colorsys
from collections import Counter
from scipy.spatial import ConvexHull
import webbrowser
import os

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance en mètres entre deux points en utilisant la formule haversine"""
    R = 6371000  # Rayon de la Terre en mètres
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    result = haversine_distances([[lat1, lon1]], [[lat2, lon2]])[0][0]
    return R * result

def get_cluster_tags(df, cluster_id, mots_exclus):
    """Trouve le tag le plus représentatif pour un cluster"""
    cluster_data = df[df['cluster'] == cluster_id]
    all_tags = []
    
    for tags_str in cluster_data['tags'].fillna(''):
        tags_list = tags_str.lower().split(',')
        for tag in tags_list:
            tag = tag.strip()
            if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                subtags = tag.replace('_', ' ').replace('-', ' ').split()
                all_tags.extend(subtags)
    
    tag_counts = Counter(all_tags)
    most_common_tags = [(tag, count) for tag, count in tag_counts.most_common(10)
                       if len(tag) > 2 and ' ' not in tag]
    
    return most_common_tags[0][0] if most_common_tags else f"Cluster{cluster_id}"

def main():
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('datasetCleaned.csv', low_memory=False)
    
    # Limiter aux coordonnées de Lyon
    lyon_bounds = {
        'lat_min': 45.73,
        'lat_max': 45.80,
        'lon_min': 4.79,
        'lon_max': 4.90
    }
    
    mask = (
        (df['lat'] >= lyon_bounds['lat_min']) &
        (df['lat'] <= lyon_bounds['lat_max']) &
        (df['long'] >= lyon_bounds['lon_min']) &
        (df['long'] <= lyon_bounds['lon_max'])
    )
    df = df[mask]
    
    # Préparer les données pour le clustering
    X = df[['lat', 'long']].values
    
    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paramètres du clustering
    n_clusters = 50
    MAX_DISTANCE = 100  # Distance maximale en mètres
    MIN_POINTS = 5     # Nombre minimum de points par cluster
    
    # Appliquer K-means
    print("Application de K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Identifier les points trop éloignés
    print("Vérification des distances...")
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        center = kmeans.cluster_centers_[cluster_id]
        center_coords = scaler.inverse_transform([center])[0]
        distance = calculate_distance(
            row['lat'], row['long'],
            center_coords[0], center_coords[1]
        )
        if distance > MAX_DISTANCE:
            df.loc[idx, 'cluster'] = -1
    
    # Définir les mots exclus pour les tags
    mots_exclus = ['unknown', 'lyon', '', 'france', 'europe', 'nuit', 'streetphotography', 'french', 'rhônealpes', 'fêtedeslumières', 'city', 'internetdesobjets']
    
    # Trouver les tags représentatifs pour chaque cluster
    cluster_tags = {}
    for cluster_id in range(n_clusters):
        if len(df[df['cluster'] == cluster_id]) >= MIN_POINTS:
            cluster_tags[cluster_id] = get_cluster_tags(df, cluster_id, mots_exclus)
    
    # Créer la carte
    print("Création de la carte...")
    m = folium.Map(location=[df['lat'].mean(), df['long'].mean()], zoom_start=13)
    
    # Ajouter le rectangle de la zone d'étude
    bounds = [[lyon_bounds['lat_min'], lyon_bounds['lon_min']],
              [lyon_bounds['lat_max'], lyon_bounds['lon_max']]]
    folium.Rectangle(bounds=bounds, color='red', weight=2, fill=False,
                    popup='Zone d\'étude').add_to(m)
    
    # Générer les couleurs pour les clusters
    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(color)
    
    # Créer les polygones des clusters
    for cluster_id in range(n_clusters):
        if cluster_id in cluster_tags:
            cluster_points = X[df['cluster'] == cluster_id]
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    polygon_points = [[cluster_points[vertex][0], cluster_points[vertex][1]]
                                   for vertex in hull.vertices]
                    
                    folium.Polygon(
                        locations=polygon_points,
                        color=colors[cluster_id],
                        weight=2,
                        fill=True,
                        fill_color=colors[cluster_id],
                        fill_opacity=0.2,
                        popup=f'<b>{cluster_tags[cluster_id]}</b>'
                    ).add_to(m)
                except Exception as e:
                    print(f"Erreur lors de la création du polygone pour le cluster {cluster_id}: {e}")
    
    # Ajouter les points
    for idx, row in df.iterrows():
        if row['cluster'] != -1 and row['cluster'] in cluster_tags:
            # Créer le lien Flickr
            flickr_link = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
            
            # Créer le popup avec le tag du cluster et le lien
            popup_content = f"""
            <div style="min-width: 200px;">
            Cluster: {cluster_tags[row['cluster']]}<br>
            <a href="{flickr_link}" target="_blank">Voir la photo sur Flickr</a>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=5,
                color=colors[row['cluster']],
                fill=True,
                popup=popup_content,
                fill_opacity=0.7
            ).add_to(m)
    
    # Sauvegarder et ouvrir la carte
    print("Sauvegarde de la carte...")
    m.save('carte_clusters.html')
    webbrowser.open('file://' + os.path.realpath('carte_clusters.html'))

if __name__ == "__main__":
    main()