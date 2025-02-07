import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import folium
import webbrowser
import os


def load_and_preprocess_data(filepath, lyon_bounds):
    """
    Charger et nettoyer les données pour le clustering.
    Filtrer les données pour réduire leur taille.
    """
    print("Chargement des données...")
    df = pd.read_csv(filepath, sep=",", low_memory=False)

    # Supprimer les espaces superflus dans les colonnes
    df.columns = df.columns.str.strip()

    # Supprimer les doublons basés sur lat/long
    df = df.drop_duplicates(subset=['lat', 'long'])
    print(f"Nombre de points après suppression des doublons : {len(df)}")

    # Filtrer les données pour garder uniquement celles situées à Lyon
    mask = (
        (df['lat'] >= lyon_bounds['lat_min']) &
        (df['lat'] <= lyon_bounds['lat_max']) &
        (df['long'] >= lyon_bounds['lon_min']) &
        (df['long'] <= lyon_bounds['lon_max'])
    )
    df = df[mask]
    print(f"Nombre de points après filtrage géographique : {len(df)}")

    # Réduire la densité en supprimant certains points (échantillonnage)
    df = df.sample(frac=0.2, random_state=42)  # Garder seulement 20% des points
    print(f"Nombre de points après échantillonnage : {len(df)}")

    return df


def apply_hierarchical_clustering(data, n_clusters):
    """
    Appliquer le clustering hiérarchique sur les données géographiques.
    """
    print("Application du clustering hiérarchique...")

    # Préparer les données pour le clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['lat', 'long']])

    # Appliquer le clustering hiérarchique
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    data['cluster'] = clustering.fit_predict(X_scaled)

    print(f"Clustering terminé. Nombre de clusters : {n_clusters}")
    return data


def visualize_clusters(data, lyon_bounds):
    """
    Visualiser les clusters sur une carte interactive avec Folium.
    """
    print("Création de la carte...")

    # Créer une carte centrée sur les données
    m = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=13)

    # Ajouter les frontières de Lyon
    bounds = [[lyon_bounds['lat_min'], lyon_bounds['lon_min']],
              [lyon_bounds['lat_max'], lyon_bounds['lon_max']]]
    folium.Rectangle(bounds=bounds, color='red', weight=2, fill=False, popup='Zone d\'étude').add_to(m)

    # Générer des couleurs uniques pour chaque cluster
    n_clusters = data['cluster'].nunique()
    colors = [
        f"#{''.join(np.random.choice(list('0123456789ABCDEF'), 6))}"
        for _ in range(n_clusters)
    ]

    # Ajouter les points des clusters
    for _, row in data.iterrows():
        cluster_id = row['cluster']
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            color=colors[cluster_id],
            fill=True,
            fill_opacity=0.5,
            popup=f"Cluster {cluster_id}"
        ).add_to(m)

    # Sauvegarder la carte
    print("Sauvegarde de la carte...")
    m.save("output/clustersHierarchical.html")
    webbrowser.open(f'file:///{os.path.abspath("output/clustersHierarchical.html")}')


def main():
    # Définir les paramètres
    data_path = "data/datasetCleaned.csv"
    lyon_bounds = {
        'lat_min': 45.709,
        'lat_max': 45.80,
        'lon_min': 4.79,
        'lon_max': 4.90
    }
    n_clusters = 25

    # Charger et préparer les données
    data = load_and_preprocess_data(data_path, lyon_bounds)

    # Appliquer le clustering hiérarchique
    clustered_data = apply_hierarchical_clustering(data, n_clusters)

    # Visualiser les clusters
    visualize_clusters(clustered_data, lyon_bounds)


if __name__ == "__main__":
    main()
