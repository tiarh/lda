import streamlit as st
import pandas as pd
from crawling import crawl_pta
import preprocessing 
import numpy as np
from vsm import vsm_term_frequency
import base64
from lda import lda_topic_modelling
from clustering import kmeans_clustering
from sklearn.feature_extraction.text import CountVectorizer
from klasifikasi import train_and_evaluate_classifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests
from bs4 import BeautifulSoup


def get_all_prodi_links():
    # URL halaman utama PTA Trunojoyo
    main_url = "https://pta.trunojoyo.ac.id/"
    
    # Lakukan permintaan ke halaman utama
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Temukan semua link prodi pada halaman utama
    prodi_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/c_search/byprod" in href:
            prodi_links.append((link.text, href))
    
    return prodi_links
# Menu 1: Data Crawling
def data_crawling():
    st.write("Data Crawling")

    # Dapatkan semua link prodi
    prodi_links = get_all_prodi_links()
    
    # Daftar pilihan prodi
    prodi_options = [prodi[0] for prodi in prodi_links]
    
    # Biarkan pengguna memilih prodi
    selected_prodi = st.selectbox("Pilih Prodi:", prodi_options)

    # Dapatkan URL prodi berdasarkan pilihan pengguna
    url_link = [prodi[1] for prodi in prodi_links if prodi[0] == selected_prodi]

    if url_link:
        id_prodi = selected_prodi

        # Tombol "Crawl" untuk memulai proses crawling
        if st.button("Crawl"):
            crawl_pta(url_link[0], id_prodi)  # Gunakan URL pertama (indeks 0) jika ada banyak URL
            st.success(f"Data Prodi '{selected_prodi}' berhasil di-crawl.")

            # Simpan hasil crawling ke dalam session state
            st.session_state.df_crawled = pd.read_csv(f'PTA_{id_prodi}.csv')
            st.write(st.session_state.df_crawled)

            # Tambahkan tombol untuk mengunduh DataFrame ke dalam format CSV
            csv_file = st.session_state.df_crawled.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()  # Encode sebagai base64
            download_filename = f'{id_prodi}_data.csv'
            href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">Download Data CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("Pilih prodi terlebih dahulu.")

def data_preprocessing():
    st.write("Data Preprocessing")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Periksa apakah data hasil crawling sudah ada dalam session state
    if 'df_crawled' in st.session_state:
        df = st.session_state.df_crawled
    else:
        df = None

    if uploaded_file is not None:
        # Jika ada file yang diunggah, gunakan data dari file
        df = pd.read_csv(uploaded_file)

    if df is not None:
        st.write("Data sebelum preprocessing:")
        st.write(df)

            # Preprocess the data
        df['Abstrak'] = df['Abstrak'].apply(lambda text: preprocessing.preprocess_text(text) if isinstance(text, str) else text)

        # Hapus baris dengan abstrak kosong
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        st.write("Data setelah preprocessing:")
        st.write(df)

        csv_file = df.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        download_filename = 'Hasil_preprocessing.csv'
        href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">Download Data CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Simpan hasil preprocessing ke dalam session state
        st.session_state.df_preprocessed = df

        # Konversi kolom Abstrak ke dalam format list teks
        texts = df['Abstrak'].tolist()

        # Panggil fungsi VSM dengan input teks
        tf_df = vsm_term_frequency(texts)

        # Simpan hasil Term Frequency ke dalam session state
        st.session_state.tf_df = tf_df

        return df



def term_frequency_analysis():
    st.write("Ekstraksi Fitur")

    # Check if preprocessing has been done previously and stored in a variable (e.g., df_preprocessed)
    if 'df_preprocessed' in st.session_state:
        df = st.session_state.df_preprocessed

        # Konversi kolom Abstrak ke dalam format list teks
        texts = df['Abstrak'].tolist()
        
        # Menggunakan CountVectorizer untuk menghitung Term Frequency
        vectorizer = CountVectorizer()
        tf_matrix = vectorizer.fit_transform(texts)

        # Membentuk DataFrame Term Frequency
        tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Menambahkan kolom "Abstrak" dengan teks abstrak asli
        tf_df.insert(0, "Judul", texts)

        st.write("Term Frequency DataFrame:")
        st.write(tf_df)
        st.write("Bentuk (Shape) dari Term Frequency DataFrame:", tf_df.shape)

    else:
        st.warning("Harap lakukan preprocessing terlebih dahulu atau pilih menu 'Data Preprocessing'.")

def topic_modelling_lda():
    try:
        st.write("LDA Topic Modelling")

        # Check if preprocessing has been done previously and stored in a variable (e.g., df_preprocessed)
        if 'df_preprocessed' in st.session_state:
            df = st.session_state.df_preprocessed

            if not df.empty:
                # Konversi kolom Abstrak ke dalam format list teks
                texts = df['Abstrak'].tolist()

                # Check if VSM has been computed previously and stored in a variable (e.g., tf_df)
                if 'tf_df' in st.session_state:
                    tf_df = st.session_state.tf_df
                else:
                    # Gunakan hasil VSM sebagai input untuk LDA
                    vectorizer = CountVectorizer()
                    tf_matrix = vectorizer.fit_transform(texts)

                    # Membentuk DataFrame Term Frequency
                    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                    # Simpan hasil VSM ke dalam session state
                    st.session_state.tf_df = tf_df

                num_topics = 5 # Slider untuk memilih jumlah topik

                U, VT_tabel, lda = lda_topic_modelling(tf_df, num_topics)

                # Tampilkan hasil LDA

                # Menambahkan kolom "Abstrak" dengan teks abstrak asli
                tf_df.insert(0, "Judul", texts)
                st.write("Term Frequency DataFrame:")
                st.write(tf_df)
                st.write(tf_df.shape)

                U.insert(0, "Judul", texts)
                st.write("Matriks Dokumen-Topik (U):")
                st.dataframe(U)
                st.session_state.U = U
                st.write(U.shape)

                st.write("Matriks Topik-Kata (VT):")
                st.dataframe(VT_tabel)
                st.write(VT_tabel.shape)
            else:
                st.warning("Data preprocessing kosong.")
        else:
            st.warning("Harap lakukan preprocessing terlebih dahulu atau pilih menu 'Data Preprocessing'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")




def clustering_kmeans():
    st.write("Clustering with K-Means")

    # Cek apakah U sudah ada di session state
    if 'U' in st.session_state:
        U = st.session_state.U

        # Melakukan clustering K-Means
        num_clusters = st.slider("Jumlah Cluster K-Means", 2, 10, 2)  # Slider untuk memilih jumlah cluster
        clusters = kmeans_clustering(U, num_clusters)

        # Menambahkan hasil clustering ke DataFrame U
        U['Cluster'] = clusters

        # Tampilkan hasil akhir dalam tabel
        st.write("Hasil Akhir:")
        st.dataframe(U)
        st.write(U.shape)

        csv_file = U.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        download_filename = 'Hasil_Clustering.csv'
        href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">Download Data CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Hitung nilai Silhouette Score
        silhouette_scores = []
        for n_clusters in range(2, num_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(U.iloc[:, 2:6])
            silhouette_avg = silhouette_score(U.iloc[:, 2:6], cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Tampilkan nilai Silhouette Score dan plot
        st.write("Nilai Silhouette Score untuk berbagai jumlah kluster:")
        for n_clusters, score in enumerate(silhouette_scores, start=2):
            st.write(f"{n_clusters}: {score}")
        st.line_chart(silhouette_scores)

        # Simpan hasil clustering ke dalam file CSV

        # Menampilkan judul-judul dalam setiap cluster
        for cluster_id in range(num_clusters):
            st.write(f"Cluster {cluster_id} - Judul:")
            cluster_titles = U[U['Cluster'] == cluster_id]['Judul']
            st.write(cluster_titles)

    else:
        st.warning("Anda perlu melakukan analisis LDA terlebih dahulu di menu 'LDA Topic Modelling'.")


# Menu 3: Klasifikasi Abstrak
def classify_abstracts():
    st.write("Klasifikasi Abstrak")

    # Tambahkan elemen upload file
    uploaded_file = st.file_uploader("Upload CSV file with labeled abstracts", type=["csv"])

    if uploaded_file is not None:
        # Jika ada file yang diunggah, baca data dari file
        labeled_data = pd.read_csv(uploaded_file)

        # Lakukan klasifikasi menggunakan modul klasifikasi
        classifier, accuracy = train_and_evaluate_classifier(labeled_data)

        # Tampilkan hasil klasifikasi
        st.write("Hasil Klasifikasi:")
        st.write(labeled_data)  # Gantilah ini dengan hasil klasifikasi yang sebenarnya

        st.write(f"Akurasi: {accuracy}")


# Main page with 4 menus
def main():
    st.title("Topic Modelling LDA")
    menu = ["Data Crawling", "Data Preprocessing", "Ekstraksi Fitur", "LDA", "Clustering with K-Means", "Klasifikasi"]
    choice = st.sidebar.selectbox("Pilih menu", menu)

    if choice == "Data Crawling":
        data_crawling()
    elif choice == "Data Preprocessing":
        data_preprocessing()
    elif choice == "Ekstraksi Fitur":
        term_frequency_analysis()
    elif choice == "LDA":
        topic_modelling_lda()
    elif choice == "Clustering with K-Means":
        clustering_kmeans()
    elif choice == "Klasifikasi":
        classify_abstracts()

if __name__ == "__main__":
    main()
