import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import nltk
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Télécharger les ressources NLTK nécessaires
nltk.download()

def preprocess_text(text, stemmer, stop_words):
    """Prétraite le texte en effectuant les opérations suivantes :
    - Mise en minuscules
    - Suppression des caractères spéciaux
    - Tokenization
    - Suppression des stopwords
    - Stemming
    """
    if isinstance(text, str):
        # Convertir en minuscules et supprimer les caractères spéciaux
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Suppression des stopwords et stemming
        tokens = [stemmer.stem(token) for token in tokens 
                 if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    return ''

def create_wordcloud(text, title):
    """Crée et affiche un nuage de mots"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def extract_frequent_itemsets(df, cluster_id):
    """Extrait les ensembles d'items fréquents pour un cluster donné"""
    # Créer une matrice binaire pour les mots
    cluster_texts = df[df['cluster'] == cluster_id]['processed_text']
    all_words = set(' '.join(cluster_texts).split())
    
    # Créer un DataFrame avec des colonnes binaires pour chaque mot
    word_matrix = pd.DataFrame()
    for word in all_words:
        word_matrix[word] = cluster_texts.apply(lambda x: 1 if word in x.split() else 0)
    
    # Appliquer l'algorithme Apriori
    frequent_itemsets = apriori(word_matrix, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    return rules

def main():
    # Charger les données
    print("Chargement des données...")
    df = pd.read_csv('data/datasetCleaned.csv', low_memory=False)
    
    # Initialiser les outils de prétraitement
    stemmer = FrenchStemmer()
    stop_words = set(stopwords.words('french') + stopwords.words('english'))
    
    # Ajouter des mots spécifiques à exclure
    additional_stops = {'lyon', 'france', 'rhone', 'rhônealpes', 'photo', 'picture', 'img', 'image'}
    stop_words.update(additional_stops)
    
    # Prétraiter les textes
    print("Prétraitement des textes...")
    df['processed_text'] = df.apply(
        lambda row: preprocess_text(
            str(row['title']) + ' ' + str(row['tags']), 
            stemmer, 
            stop_words
        ), 
        axis=1
    )
    
    # Analyser chaque cluster
    unique_clusters = df['cluster'].unique()
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Ignorer les points de bruit
            continue
            
        print(f"\nAnalyse du cluster {cluster_id}")
        cluster_texts = df[df['cluster'] == cluster_id]['processed_text']
        
        # 1. TF-IDF
        print("Calcul du TF-IDF...")
        tfidf = TfidfVectorizer(max_features=10)
        tfidf_matrix = tfidf.fit_transform(cluster_texts)
        
        # Obtenir les mots les plus importants selon TF-IDF
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.mean(axis=0).A1
        top_words = [(feature_names[i], score) for i, score in enumerate(tfidf_scores)]
        top_words.sort(key=lambda x: x[1], reverse=True)
        
        print("Mots les plus significatifs (TF-IDF):")
        for word, score in top_words[:5]:
            print(f"  - {word}: {score:.4f}")
        
        # 2. Créer un nuage de mots
        all_text = ' '.join(cluster_texts)
        create_wordcloud(all_text, f'Nuage de mots - Cluster {cluster_id}')
        
        # 3. Règles d'association
        print("Calcul des règles d'association...")
        rules = extract_frequent_itemsets(df, cluster_id)
        if not rules.empty:
            print("\nRègles d'association les plus pertinentes:")
            top_rules = rules.nlargest(5, 'lift')
            for _, rule in top_rules.iterrows():
                print(f"  {rule['antecedents']} => {rule['consequents']}")
                print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")

if __name__ == "__main__":
    main() 