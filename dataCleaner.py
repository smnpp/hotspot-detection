# load pandas to deal with the data
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load folium to create maps
import folium
from folium import Map
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from sklearn.preprocessing import StandardScaler

# load data from table file where entries are separated with a ','
data = pd.read_table("dataset.csv", sep=",", low_memory=False)

# remove leading and trailing whitespaces from column names
data.columns = data.columns.str.strip()
# check the columns names
print(data.columns)
# check the data types of the columns
print(data.info())
# check the first few rows of the data
data.head()

# coordonnées géographiques retenu pour Lyon = [45.75, 4.85]
# rayon de 15 km autour de ce point
validation_rules = {
    'lat': lambda x: pd.api.types.is_number(x) and 45.614067767464974 <= x <= 45.88380569722158,
    'long': lambda x: pd.api.types.is_number(x) and 4.655505238288724 <= x <= 5.042868327562071,
    'date_taken_minute': lambda x: pd.api.types.is_number(x) and 0 <= x <= 59,
    'date_taken_hour': lambda x: pd.api.types.is_number(x) and 0 <= x <= 23,
    'date_taken_day': lambda x: pd.api.types.is_number(x) and 1 <= x <= 31,
    'date_taken_month': lambda x: pd.api.types.is_number(x) and 1 <= x <= 12,
    'date_taken_year': lambda x: pd.api.types.is_number(x) and 1839 <= x <= 2024,
    'date_upload_minute': lambda x: pd.api.types.is_number(x) and 0 <= x <= 59,
    'date_upload_hour': lambda x: pd.api.types.is_number(x) and 0 <= x <= 23,
    'date_upload_day': lambda x: pd.api.types.is_number(x) and 1 <= x <= 31,
    'date_upload_month': lambda x: pd.api.types.is_number(x) and 1 <= x <= 12,
    'date_upload_year': lambda x: pd.api.types.is_number(x) and 1839 <= x <= 2024,
}

# Fonction de nettoyage des colonnes
def clean_column(dataframe, column_name, validation_func):
    dataframe[column_name] = dataframe[column_name].apply(
        lambda x: x if validation_func(x) else np.nan
    )

# Appliquer les règles de validation à chaque colonne
for column, rule in validation_rules.items():
    if column in data.columns:
        clean_column(data, column, rule)

print(f"Before removing missing values: {len(data)}")
# remove rows with missing values on the columns id, lat, and long
data_cleaned_mv = data.dropna(subset=['id', 'lat', 'long'])
print(f"After removing missing values: {len(data_cleaned_mv)}")

print(f"Before removing exact duplicates: {len(data_cleaned_mv)}")

# Supprimer uniquement les doublons exacts (id, user, lat, long)
data_cleaned_d = data_cleaned_mv.drop_duplicates(subset=['id', 'user', 'lat', 'long'], keep='first')

# Afficher la taille après suppression des doublons
print(f"After removing exact duplicates: {len(data_cleaned_d)}")

data_cleaned_d.to_csv('datasetCleaned.csv', index=False)
print("Données nettoyées sauvegardées dans 'datasetCleaned.csv'")
