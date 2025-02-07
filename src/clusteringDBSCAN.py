import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import folium
from scipy.spatial import ConvexHull
import os
import webbrowser


def load_and_preprocess_data(filepath, sample_size=10000):
    """
    Charger et nettoyer les données pour DBSCAN.
    """
    print("Chargement des données...")
    df = pd.read_csv(filepath, low_memory=False)

    # Limiter à une taille spécifique pour améliorer les performances
    df = df.head(sample_size)
    print(f"Taille du dataset : {len(df)}")

    # Supprimer les doublons basés sur lat/long
    df = df.drop_duplicates(subset=['lat', 'long'])
    print(f"Nombre de points après suppression des doublons : {len(df)}")

    return df


def apply_dbscan(data, eps, min_samples):
    """
    Appliquer DBSCAN pour le clustering.
    """
    print("Application de DBSCAN...")

    # Préparer les données géographiques pour le clustering
    X = data[['lat', 'long']].values

    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['cluster'] = dbscan.fit_predict(X)

    # Calculer les statistiques des clusters
    n_clusters = len(set(data['cluster'])) - (1 if -1 in data['cluster'] else 0)
    n_noise = sum(data['cluster'] == -1)
    print(f"Nombre de clusters trouvés : {n_clusters}")
    print(f"Nombre de points de bruit : {n_noise}")

    return data, n_clusters


def visualize_clusters(data, n_clusters):
    """
    Visualiser les clusters sur une carte interactive avec Folium.
    """
    print("Création de la carte...")

    # Créer une carte centrée sur les données
    m = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=12)

    # Générer des couleurs uniques pour chaque cluster
    colors = ['#808080']  # Couleur grise pour le bruit
    colors += [
        f"#{''.join(np.random.choice(list('0123456789ABCDEF'), 6))}"
        for _ in range(n_clusters)
    ]

    # Ajouter les points des clusters
    for _, row in data.iterrows():
        cluster_id = row['cluster']
        color_idx = cluster_id + 1 if cluster_id >= 0 else 0  # Couleur grise pour le bruit
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=colors[color_idx],
            fill=True,
            fill_opacity=0.5 if cluster_id >= 0 else 0.2
        ).add_to(m)

    # Ajouter les polygones convexes pour les clusters
    for cluster_id in range(n_clusters):
        cluster_points = data[data['cluster'] == cluster_id][['lat', 'long']].values
        if len(cluster_points) >= 3:  # Un polygone nécessite au moins 3 points
            try:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                folium.Polygon(
                    locations=[[point[0], point[1]] for point in hull_points],
                    color=colors[cluster_id + 1],
                    weight=2,
                    fill=True,
                    fill_color=colors[cluster_id + 1],
                    fill_opacity=0.2
                ).add_to(m)
            except Exception as e:
                print(f"Erreur lors de la création du polygone pour le cluster {cluster_id}: {e}")

    # Sauvegarder la carte
    print("Sauvegarde de la carte...")
    m.save("output/clusteringDBSCAN.html")
    webbrowser.open(f'file:///{os.path.abspath("output/clusteringDBSCAN.html")}')


def main():
    # Paramètres
    data_path = "data/datasetCleaned.csv"
    sample_size = 10000  # Limite des points pour améliorer les performances
    eps = 0.0004  # Rayon de voisinage
    min_samples = 4  # Nombre minimum de points pour former un cluster

    # Charger et préparer les données
    df = load_and_preprocess_data(data_path, sample_size)

    # Appliquer DBSCAN
    clustered_data, n_clusters = apply_dbscan(df, eps, min_samples)

    # Visualiser les clusters
    visualize_clusters(clustered_data, n_clusters)


if __name__ == "__main__":
    main()