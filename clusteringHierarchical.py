#imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import folium
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import colorsys
from collections import Counter
import webbrowser
import os

# load data from table file where entries are separated with a ','
dataHierarch = pd.read_table("datasetCleaned.csv", sep=",", low_memory=False)

# remove leading and trailing whitespaces from column names
dataHierarch.columns = dataHierarch.columns.str.strip()
#remove entries with same coordinates
print(len(dataHierarch))
dataHierarch=dataHierarch.drop_duplicates(subset=['lat', 'long'])
print(len(dataHierarch))

# Limiter aux coordonnées de Lyon et diviser dans deux zones geographiques
lyon_bounds = {
    'lat_min': 45.709, #original: 45.709
    'lat_max': 45.762, #original: 45.80
    'lon_min': 4.79, #original:  4.79
    'lon_max': 4.90 #original: 4.90
}

mask = (
    (dataHierarch['lat'] >= lyon_bounds['lat_min']) &
    (dataHierarch['lat'] <= lyon_bounds['lat_max']) &
    (dataHierarch['long'] >= lyon_bounds['lon_min']) &
    (dataHierarch['long'] <= lyon_bounds['lon_max'])
)
dataHierarch1 = dataHierarch[mask]
# Indizes von dataHierarch1 erhalten
indices_to_exclude = dataHierarch1.index

# Zeilen auswählen, die nicht in den Indizes von dataHierarch1 enthalten sind
dataHierarch2 = dataHierarch.loc[~dataHierarch.index.isin(indices_to_exclude)]
print(len(dataHierarch))
print(len(dataHierarch1))
print(len(dataHierarch2))

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


def cluster_and_visualize(data1, data2, lyon_bounds, mots_exclus):
    # Daten vorbereiten
    data1 = data1[['lat', 'long', 'tags']].copy()
    data2 = data2[['lat', 'long', 'tags']].copy()

    # Clustering für beide Datensets durchführen
    scaler = StandardScaler()
    clusters_per_set = 25

    # Erstes Datenset clustern
    X1 = scaler.fit_transform(data1[['lat', 'long']])
    cluster_model1 = AgglomerativeClustering(n_clusters=clusters_per_set, linkage='ward')
    data1['cluster'] = cluster_model1.fit_predict(X1)

    # Zweites Datenset clustern
    X2 = scaler.fit_transform(data2[['lat', 'long']])
    cluster_model2 = AgglomerativeClustering(n_clusters=clusters_per_set, linkage='ward')
    data2['cluster'] = cluster_model2.fit_predict(X2)

    # Cluster-IDs des zweiten Datensets anpassen, um IDs 0-49 zu haben
    data2['cluster'] += clusters_per_set

    # Beide Datensets zusammenfügen
    data_combined = pd.concat([data1, data2], ignore_index=True)

    # Cluster-Tags berechnen
    cluster_tags = {}
    for cluster_id in range(50):
        cluster_tags[cluster_id] = get_cluster_tags(data_combined, cluster_id, mots_exclus)

    representative_points = (
    data_combined.groupby('cluster')[['lat', 'long']].mean().reset_index()
    )


    # Karte erstellen
    print("Erstellen der Karte...")
    m = folium.Map(location=[data_combined['lat'].mean(), data_combined['long'].mean()], zoom_start=13)

    # Rechteck der Lyon-Grenzen hinzufügen
    bounds = [[lyon_bounds['lat_min'], lyon_bounds['lon_min']],
              [lyon_bounds['lat_max'], lyon_bounds['lon_max']]]
    folium.Rectangle(bounds=bounds, color='red', weight=2, fill=False, popup='Zone d\'étude').add_to(m)

    # Farben für Cluster generieren
    colors = []
    for i in range(50):
        hue = i / 50
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        colors.append(color)

       # Repräsentative Punkte auf die Karte setzen
    for _, row in representative_points.iterrows():
        cluster_id = int(row['cluster'])
        popup_content = f"""
        <div style="min-width: 200px;">
        Cluster: {cluster_tags[cluster_id]}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=8,
            color=colors[cluster_id],
            fill=True,
            popup=popup_content,
            fill_opacity=0.7
        ).add_to(m)

    # Karte speichern und öffnen
    print("Karte speichern...")
    m.save('carte_clusters.html')
    webbrowser.open(f'file:///{os.path.abspath("carte_clusters.html")}')

    return data1, data2


# Beispiel-Daten und Parameter
lyon_bounds = {
    'lat_min': 45.709,
    'lat_max': 45.80,
    'lon_min': 4.79,
    'lon_max': 4.90
}

mots_exclus = ['unknown', 'lyon', '', 'france', 'europe', 'nuit', 'streetphotography', 'french', 'rhônealpes']

# Hauptfunktion ausführen
data_hierarchical1, data_hierarchical2 = cluster_and_visualize(dataHierarch1, dataHierarch2, lyon_bounds, mots_exclus)
