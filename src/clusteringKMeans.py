import pandas as pd
import numpy as np
import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    """
    Charge et nettoie les données pour le clustering.
    Filtre les données pour garder uniquement celles situées à Lyon.
    """
    print("Chargement des données...")
    df = pd.read_csv(filepath, low_memory=False)

    # Filtrer les données pour garder uniquement celles situées à Lyon
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
    return df

def apply_kmeans(df, n_clusters=50):
    """
    Applique l'algorithme K-Means sur les données de latitude et longitude.
    Retourne le DataFrame avec les clusters attribués et les centres.
    """
    print("Application de K-means...")

    # Préparer les données pour le clustering
    X = df[['lat', 'long']].values

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler

def visualize_clusters(df, kmeans, output_file="output/clusteringKMeans.html"):
    """
    Génère une carte Folium affichant les clusters et les centres.
    """
    print("Création de la carte Folium...")

    # Initialiser la carte centrée sur Lyon
    map_clusters = folium.Map(location=[45.75, 4.85], zoom_start=13)

    # Générer une couleur unique par cluster
    cluster_colors = [
        f"#{''.join(np.random.choice(list('0123456789ABCDEF'), 6))}"
        for _ in range(len(df['cluster'].unique()))
    ]

    # Ajouter les points des clusters sur la carte
    for idx, row in df.iterrows():
        cluster_id = row['cluster']
        color = cluster_colors[cluster_id]
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.5
        ).add_to(map_clusters)

    # Ajouter les centres des clusters
    for i, center in enumerate(kmeans.cluster_centers_):
        center_coords = center[::-1]  # Inversion lat/lon
        folium.Marker(
            location=center_coords,
            popup=f"Cluster Center {i}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(map_clusters)

    # Sauvegarde et affichage
    map_clusters.save(output_file)
    print(f"Carte enregistrée : {output_file}")

def main():
    # Charger les données nettoyées
    data_path = "data/datasetCleaned.csv"
    df = load_and_preprocess_data(data_path)

    # Appliquer K-Means
    df, kmeans, scaler = apply_kmeans(df, n_clusters=50)

    # Sauvegarder les résultats
    output_path = "output/clusters_kmeans.csv"
    df.to_csv(output_path, index=False)
    print(f"Résultats sauvegardés dans {output_path}")

    # Générer et afficher la carte
    visualize_clusters(df, kmeans)

if __name__ == "__main__":
    main()
