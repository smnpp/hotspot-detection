import pandas as pd
import folium
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from scipy.spatial import ConvexHull
import colorsys
from collections import Counter
import webbrowser
import os

def main():
    global df, clustering_algo, N
    
    # Préparer les données pour la clusterisation
    X = df[['lat', 'long']].values
    
    # Appliquer l'algorithme de clustering
    df['cluster'] = clustering_algo.fit_predict(X)
    
    # Si K-means est utilisé, les clusters commencent à 0 et sont tous positifs
    # Pour DBSCAN, -1 représente le bruit
    
    # Trouver les tags les plus communs dans tout le dataset
    all_dataset_tags = []
    for tags_str in df['tags'].fillna(''):
        tags_list = tags_str.lower().split(',')
        for tag in tags_list:
            tag = tag.strip()
            mots_exclus = ['unknown', 'lyon', '', 'france', 'europe']
            if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                subtags = tag.replace('_', ' ').replace('-', ' ').split()
                all_dataset_tags.extend(subtags)

    # Trouver les N tags les plus communs dans tout le dataset
    common_tags = [tag for tag, _ in Counter(all_dataset_tags).most_common(N)]
    print("Tags les plus communs exclus:", common_tags)

    # Ajouter ces tags communs à la liste des mots exclus
    mots_exclus = ['unknown', 'lyon', '', 'france', 'europe','nuit','streetphotography','french']
    mots_exclus.extend(common_tags)
    print("Mots exclus:", mots_exclus)

    # Trouver le tag le plus représentatif pour chaque cluster
    cluster_tags = {}
    unique_clusters = sorted(df['cluster'].unique())

    for cluster_id in unique_clusters:
        # Obtenir tous les tags du cluster
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Créer une liste de tous les tags du cluster
        all_tags = []
        for tags_str in cluster_data['tags'].fillna(''):
            tags_list = tags_str.lower().split(',')
            for tag in tags_list:
                tag = tag.strip()
                if tag not in mots_exclus and not any(c.isdigit() for c in tag):
                    subtags = tag.replace('_', ' ').replace('-', ' ').split()
                    all_tags.extend(subtags)
        
        # Compter les occurrences
        tag_counts = Counter(all_tags)
        
        # Trouver le tag le plus fréquent (en excluant les tags trop courts)
        most_common_tags = [(tag, count) for tag, count in tag_counts.most_common(10)
                           if len(tag) > 2 and ' ' not in tag]  # Ignorer les tags trop courts et ceux avec des espaces
        
        if most_common_tags:
            cluster_tags[cluster_id] = most_common_tags[0][0]
        else:
            cluster_tags[cluster_id] = f"Cluster{cluster_id}"

    # Nombre de clusters trouvés (excluant le bruit qui est -1)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"Nombre de clusters trouvés : {n_clusters}")

    # Générer des couleurs aléatoires pour chaque cluster
    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(color)
    # Ajouter une couleur grise pour les points de bruiti
    colors = ['#808080'] + colors

    # Créer une carte centrée sur la moyenne des coordonnées
    carte = folium.Map(
        location=[df['lat'].mean(), df['long'].mean()],
        zoom_start=12
    )

    # Ajouter le rectangle englobant
    bounds = [
        [df['lat'].min(), df['long'].min()],  # coin sud-ouest
        [df['lat'].max(), df['long'].max()]   # coin nord-est
    ]
    print(bounds)

    bounds = [[45.73, 4.79], [45.80, 4.90]]
    
    folium.Rectangle(
        bounds=bounds,
        color='red',
        weight=2,
        fill=False,
        popup='Zone d\'étude',
        opacity=0.7
    ).add_to(carte)

    # Créer les zones de clusters avant d'ajouter les points
    for cluster_id in range(n_clusters):
        mask = df['cluster'] == cluster_id
        cluster_points = X[mask]
        
        # Vérifier qu'il y a assez de points différents pour former un polygone
        unique_points = np.unique(cluster_points, axis=0)
        if len(unique_points) >= 3:
            try:
                # Ajouter un petit bruit aléatoire pour éviter les points coplanaires
                jittered_points = unique_points + np.random.normal(0, 1e-10, unique_points.shape)
                hull = ConvexHull(jittered_points)
                hull_points = jittered_points[hull.vertices]
                polygon_points = [[point[0], point[1]] for point in hull_points]
                
                # Utiliser le tag le plus représentatif dans le popup
                cluster_name = cluster_tags[cluster_id]
                folium.Polygon(
                    locations=polygon_points,
                    color=colors[cluster_id + 1],
                    weight=2,
                    fill=True,
                    fill_color=colors[cluster_id + 1],
                    fill_opacity=0.2,
                    popup=f'<b>{cluster_name}</b>'
                ).add_to(carte)
            except Exception as e:
                print(f"Impossible de créer le polygone pour le cluster {cluster_id}: {e}")

    # Ajouter chaque point à la carte avec la couleur de son cluster
    for idx, row in df.iterrows():
        color_idx = row['cluster'] + 1 if row['cluster'] >= 0 else 0
        
        # Utiliser le tag le plus représentatif dans le popup
        cluster_name = cluster_tags[row['cluster']] if row['cluster'] >= 0 else "Non clustérisé"
        
        # Créer le lien Flickr
        flickr_link = f"https://www.flickr.com/photos/{row['user']}/{row['id']}"
        
        # Créer le popup avec le lien HTML et un style pour une largeur fixe
        popup_content = f"""
        <div style="min-width: 200px;">
        Cluster: {cluster_name}<br>
        <a href="{flickr_link}" target="_blank">Voir la photo sur Flickr</a>
        </div>
        """
        
        # Ajuster la taille et l'opacité selon que le point est dans un cluster ou non
        radius = 5 if row['cluster'] >= 0 else 3
        opacity = 0.7 if row['cluster'] >= 0 else 0.15
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=radius,  # Plus petit pour les points non clusterisés
            color=colors[color_idx],
            fill=True,
            popup=popup_content,
            fill_opacity=opacity  # Plus transparent pour les points non clusterisés
        ).add_to(carte)

    # Sauvegarder la carte en HTML
    carte.save('carte_photos.html')
    webbrowser.open('file://' + os.path.realpath('carte_photos.html'))

if __name__ == "__main__":
    # Configuration par défaut si exécuté directement
    df = pd.read_csv('datasetCleaned.csv', low_memory=False)
    df = df.head(10000)
    print(f"taille du dataset: {len(df)}")
    clustering_algo = DBSCAN(eps=0.0003, min_samples=5)
    N = 100
    main() 