import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Data Clean
def clean(data):
    # HTML Tag Removal
    data = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(data))

    # Case folding
    data = data.lower()

    # Trim data
    data = data.strip()

    # Remove punctuations, karakter spesial, and spasi ganda
    data = re.compile('<.*?>').sub('', data)
    data = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', data)
    data = re.sub('\s+', ' ', data)

    # Number removal
    data = re.sub(r'\[[0-9]*\]', ' ', data)
    data = re.sub(r'[^\w\s]', '', str(data).lower().strip())
    data = re.sub(r'\d', ' ', data)
    data = re.sub(r'\s+', ' ', data)

    # Mengubah data 'nan' dengan whitespace agar nantinya dapat dihapus
    data = re.sub('nan', '', data)

    return data

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokens if word.lower() not in stop_words]

def preprocess_text(text):
    if isinstance(text, str):
        text = clean(text)
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        if not tokens:  # Cek jika tidak ada token setelah pembersihan
            return np.nan
        return ' '.join(tokens)
    else:
        return np.nan  # Ubah nilai selain tipe data string menjadi NaN


def remove_missing_values(df):
    # Hapus baris dengan nilai-nilai yang hilang (NaN)
    df.dropna(inplace=True)
    return df
