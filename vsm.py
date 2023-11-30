import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def vsm_term_frequency(texts):
    # Menggunakan CountVectorizer untuk menghitung Term Frequency
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(texts)

    # Membentuk DataFrame Term Frequency
    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tf_df