from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

def kmeans_clustering(U, num_clusters=2):
    # Pastikan U adalah matriks 2D yang valid
    if not isinstance(U, pd.DataFrame):
        raise ValueError("U harus berupa pandas DataFrame.")
    
    # Kolom-kolom yang akan digunakan untuk clustering (Topik 1, Topik 2, Topik 3)
    cluster_data = U.iloc[:, 2:6]  # Anda harus sesuaikan indeks kolomnya dengan urutan yang benar

    # Inisialisasi model K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Melakukan clustering
    clusters = kmeans.fit_predict(cluster_data)

    return clusters
