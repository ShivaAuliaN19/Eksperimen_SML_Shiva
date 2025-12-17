print("=== FILE TERBARU DIJALANKAN ===")

import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split

try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

def preprocessing_pipeline(csv_path, kamus_path, lexicon_pos_path, lexicon_neg_path):
    df = pd.read_csv(csv_path)
    positive_lexicon = set(pd.read_csv(lexicon_pos_path, header=None)[0])
    negative_lexicon = set(pd.read_csv(lexicon_neg_path, header=None)[0])
    print("CSV berhasil dibaca")
    print("Jumlah baris :", len(df))
    print("Masuk fungsi preprocessing_pipeline")
    print("CSV path   :", csv_path)
    print("Kamus path :", kamus_path)


    #menghapus data yang duplikat
    df.drop_duplicates(subset ="content", keep = 'first', inplace = True)

    #proses cleaning data
    def remove_URL(tweet):
        if tweet is not None and isinstance(tweet, str):
            html = re.compile(r'<.*?>')
            return html.sub(r'', tweet)
        else:
            return tweet
    def remove_emoji(tweet):
        if tweet is not None and isinstance(tweet, str):
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F700-\U0001F77F"  # alchemical symbols
                u"\U0001F780-\U0001F7FF"  # geometric shapes extended
                u"\U0001F800-\U0001F8FF"  # supplemental arrows-C
                u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                u"\U0001FA00-\U0001FAFF"  # chess symbols + pictographs extended
                u"\U00002700-\U000027BF"  # dingbats
                u"\U000024C2-\U0001F251"  # enclosed characters
                u"\U0001F004-\U0001F0CF"  # additional emoticons
                u"\U0001F1E0-\U0001F1FF"  # flags
                                    "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', tweet)
        else:
            return tweet
    def remove_symbols(tweet):
        if tweet is not None and isinstance(tweet, str):
            tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
        return tweet
    def remove_numbers(tweet):
        if tweet is not None and isinstance(tweet, str):
            tweet = re.sub(r'\d', '', tweet)
        return tweet
    df['cleaning'] = df['content'].apply(lambda x: remove_URL(x))
    df['cleaning'] = df['cleaning'].apply(lambda x: remove_emoji(x))
    df['cleaning'] = df['cleaning'].apply(lambda x: remove_symbols(x))
    df['cleaning'] = df['cleaning'].apply(lambda x: remove_numbers(x))

    #proses case folding
    def case_folding(tweet):
        if isinstance(tweet, str):
            lowercase_text = tweet.lower()
            return lowercase_text
        else:
            return tweet
    df['case_folding'] = df['cleaning'].apply(case_folding)

    #proses normalisasi data
    def replace_taboo_words(text, kamus_tidak_baku):
        if isinstance(text, str):
            words = text.split()
            replaced_words = []
            kalimat_baku = []
            kata_diganti = []
            kata_tidak_baku_hash = []

            for word in words:
                if word in kamus_tidak_baku:
                    baku_word = kamus_tidak_baku[word]
                    if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                        replaced_words.append(baku_word)
                        kalimat_baku.append(baku_word)
                        kata_diganti.append(word)
                        kata_tidak_baku_hash.append(hash(word))
                else:
                    replaced_words.append(word)
            replaced_text = ' '.join(replaced_words)
        else:
            replaced_text = ''
            kalimat_baku = []
            kata_diganti = []
            kata_tidak_baku_hash = []
        return replaced_text, kalimat_baku, kata_diganti, kata_tidak_baku_hash
    data = pd.DataFrame(df[['at','userName','score','content', 'cleaning', 'case_folding']])
    kamus_data = pd.read_excel(kamus_path)
    kamus_data = kamus_data.dropna()
    kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
    data['normalisasi'], data['Kata_Baku'], data['Kata_Tidak_Baku'], data['Kata_Tidak_Baku_Hash'] = zip(
        *data['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
    )
    df = pd.DataFrame(
        data[['at', 'userName', 'score', 'content', 'cleaning', 'case_folding', 'normalisasi']]
    )

    #proses tokenization
    def tokenization(text):
        tokens = text.split()
        return tokens
    df['tokenize'] = df['normalisasi'].apply(tokenization)

    #proses stopword removal
    stop_words = stopwords.words('indonesian')
    def remove_stopwords(text):
        return [word for word in text if word not in stop_words]
    df['stopword_removal'] = df['tokenize'].apply(lambda x: remove_stopwords(x))

    #proses steming data
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stem_text(text):
        return [stemmer.stem(word) for word in text]

    df['steming_data'] = df['stopword_removal'].apply(lambda x: ' ' .join(stem_text(x)))

    #proses deteksi dan penanganan outlier
    df["content_length"] = df["cleaning"].apply(len)

    Q1 = df["content_length"].quantile(0.25)
    Q3 = df["content_length"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df["content_length"] >= lower) & (df["content_length"] <= upper)]
    print("Setelah outlier removal:", df.shape)
 
    def determine_sentiment(text):
        if not isinstance(text, str):
            return "Netral"

        positive_count = sum(1 for word in text.split() if word in positive_lexicon)
        negative_count = sum(1 for word in text.split() if word in negative_lexicon)

        if positive_count > negative_count:
            return "Positif"
        elif positive_count < negative_count:
            return "Negatif"
        else:
            return "Netral"

    df["Sentiment"] = df["steming_data"].apply(determine_sentiment)
    print("Setelah sentiment labeling:", df.shape)


    #proses pembagian data train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        df['steming_data'],
        df['Sentiment'],
        test_size=0.2,
        random_state=42
    )
    train_set = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    train_set.to_csv('train_data.csv', index=False)
    test_set = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    test_set.to_csv('test_data.csv', index=False)
    print(f'Jumlah Data Latih: {len(X_train)}')
    print(f'Jumlah Data Uji: {len(X_test)}')

    output_path = "Hasil_Preprocessing_dan_Labelling.csv"
    if df.empty:
        print("WARNING: DataFrame kosong, file tidak dibuat")
    else:
        df.to_csv(output_path, index=False)
        print("FILE DISIMPAN DI:", os.path.abspath(output_path))


if __name__ == "__main__":
    print("SCRIPT DIMULAI")

    preprocessing_pipeline(
    "dataset_raw/hasil_scraper_review_app_duolingo.csv",
    "dataset_raw/kamuskatabaku.xlsx",
    "dataset_raw/lexicon_positif.csv",
    "dataset_raw/lexicon_negatif.csv"
    )
  

    print("SCRIPT SELESAI")




