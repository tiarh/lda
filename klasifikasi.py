import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate_classifier(data):
    # Membagi data menjadi data latih dan data uji
    X = data['Abstrak']
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mengubah teks menjadi vektor fitur dengan CountVectorizer
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Melatih model klasifikasi Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Melakukan prediksi pada data uji
    y_pred = classifier.predict(X_test)

    # Mengukur akurasi
    accuracy = accuracy_score(y_test, y_pred)
    return classifier, accuracy
