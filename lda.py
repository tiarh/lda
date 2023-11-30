import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st

def lda_topic_modelling(tf_df, num_topics=3):
    # Pastikan tf_df adalah matriks 2D yang valid
    if not isinstance(tf_df, pd.DataFrame):
        raise ValueError("tf_df harus berupa pandas DataFrame.")
    
    # Pastikan tf_df tidak mengandung nilai NaN
    if tf_df.isna().any().any():
        raise ValueError("tf_df tidak boleh berisi nilai NaN.")
    
    # Menginisialisasi model LDA
    lda = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
    
    # Melakukan analisis LDA pada data Term Frequency
    lda_top = lda.fit_transform(tf_df)

    # Bobot setiap topik terhadap dokumen
    U = pd.DataFrame(lda_top, columns=[f'Topik {i + 1}' for i in range(num_topics)])

    # Bobot setiap kata terhadap topik
    VT_tabel = pd.DataFrame(lda.components_, columns=tf_df.columns)
    
    return U, VT_tabel, lda # Return the lda_top matrix as well
