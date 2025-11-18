import pandas as pd
import glob
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Abaikan peringatan untuk output yang bersih
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. Muat dan Gabungkan Data ---
print("Membaca semua file CSV...")
csv_files = glob.glob('Copy of data_pasien fix.xlsx - *.csv')
all_dfs = []

for f in csv_files:
    try:
        df = pd.read_csv(f, delimiter=';')
        df.columns = df.columns.str.strip() # Membersihkan spasi di nama kolom
        all_dfs.append(df)
    except Exception as e:
        print(f"Tidak bisa membaca {f}: {e}")

if not all_dfs:
    print("Error: Tidak ada file CSV yang ditemukan. Pastikan nama file sudah benar.")
else:
    data = pd.concat(all_dfs, ignore_index=True)
    print(f"Sukses menggabungkan {len(all_dfs)} file. Total data: {len(data)} baris.")

# --- 2. Fungsi Pra-pemrosesan Data ---
def clean_lama_penggunaan(text):
    """Mengambil angka dari teks 'Lama Penggunaan' (misal: '8 Tahun' -> 8)"""
    if isinstance(text, str):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
    return np.nan # Kembalikan NaN jika tidak ada angka

def preprocess_data(df):
    """Fungsi lengkap untuk membersihkan, mengubah, dan menyiapkan data"""
    print("Memulai pra-pemrosesan data...")
    
    # Pilih hanya kolom yang kita butuhkan
    required_cols = ['Jenis Kelamin', 'Jenis NAPZA yang digunakan (dapat lebih dari 1)', 'Lama Penggunaan NAPZA', 'Tingkat Keparahan']
    df_clean = df[required_cols].copy()
    
    df_clean.rename(columns={
        'Jenis NAPZA yang digunakan (dapat lebih dari 1)': 'Jenis_NAPZA',
        'Lama Penggunaan NAPZA': 'Lama_Penggunaan'
    }, inplace=True)
    
    # Bersihkan data 'Lama_Penggunaan'
    df_clean['Lama_Penggunaan_Clean'] = df_clean['Lama_Penggunaan'].apply(clean_lama_penggunaan)
    
    # Buang baris data yang kolom pentingnya kosong
    df_clean.dropna(subset=['Jenis_NAPZA', 'Lama_Penggunaan_Clean', 'Tingkat Keparahan', 'Jenis Kelamin'], inplace=True)
    
    # --- PERBAIKAN DATA TIDAK SEIMBANG ---
    # Kita gabungkan kelas 'Sangat Berat' ke 'Berat' karena jumlahnya terlalu sedikit
    df_clean['Tingkat Keparahan'] = df_clean['Tingkat Keparahan'].str.strip().replace('Sangat Berat', 'Berat')
    print(f"Kelas target baru: {df_clean['Tingkat Keparahan'].unique()}")
    
    # a) Encode 'Jenis Kelamin' (Laki-Laki/Perempuan -> 1/0)
    df_clean['Jenis Kelamin'] = df_clean['Jenis Kelamin'].str.strip()
    valid_genders = ['Perempuan', 'Laki-Laki']
    df_clean = df_clean[df_clean['Jenis Kelamin'].isin(valid_genders)].copy()
    
    ordinal_encoder = OrdinalEncoder(categories=[['Perempuan', 'Laki-Laki']])
    df_clean['Jenis_Kelamin_Encoded'] = ordinal_encoder.fit_transform(df_clean[['Jenis Kelamin']])
    
    # b) Encode 'Jenis_NAPZA' (Teks -> One-Hot Vector)
    # misal: "Shabu; Ganja" -> [shabu: 1, ganja: 1, heroin: 0]
    df_clean['Jenis_NAPZA_Clean'] = df_clean['Jenis_NAPZA'].str.replace(';', ' ').str.replace(',', ' ').str.replace(r'\s+', ' ', regex=True).str.lower()
    napza_vectorizer = CountVectorizer(token_pattern=r'[a-zA-Z0-9]+', min_df=2) # min_df=2 untuk abaikan kata yg terlalu jarang
    napza_features = napza_vectorizer.fit_transform(df_clean['Jenis_NAPZA_Clean']).toarray()
    napza_feature_names = napza_vectorizer.get_feature_names_out()
    
    # c) Scale 'Lama_Penggunaan' (Numerik -> 0 s/d 1)
    # Ini wajib untuk KNN
    scaler = MinMaxScaler()
    df_clean['Lama_Penggunaan_Scaled'] = scaler.fit_transform(df_clean[['Lama_Penggunaan_Clean']])
    
    # --- 4. Siapkan Fitur (X) dan Target (y) ---
    X_base_features = df_clean[['Jenis_Kelamin_Encoded', 'Lama_Penggunaan_Scaled']]
    napza_df = pd.DataFrame(napza_features, columns=napza_feature_names, index=X_base_features.index)
    X_all_features = pd.concat([X_base_features, napza_df], axis=1)
    
    # Encode Target 'Tingkat Keparahan' (Berat/Sedang/Ringan -> 0/1/2)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean['Tingkat Keparahan'])
    
    print("Pra-pemrosesan selesai.")
    # Kembalikan semua yang kita butuhkan untuk melatih DAN memprediksi
    return X_all_features, y, ordinal_encoder, napza_vectorizer, scaler, label_encoder

# --- 5. Jalankan Proses ---
if __name__ == "__main__":
    X, y, ordinal_enc, napza_vec, num_scaler, label_enc = preprocess_data(data)

    print("\nFitur yang digunakan untuk model:")
    print(X.columns.tolist())
    
    print("\nKelas target yang akan diprediksi:")
    print(list(label_enc.classes_))

    # --- 6. Pisahkan Data Latih & Uji ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData dibagi: {len(X_train)} data latih, {len(X_test)} data uji.")

    # --- 7. Latih Model ---
    
    # a) K-Nearest Neighbors (KNN)
    print("\nMelatih model KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=5) # k=5 adalah nilai default yang baik
    knn_model.fit(X_train, y_train)
    
    # b) Naive Bayes (Multinomial)
    print("Melatih model Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    print("Pelatihan model selesai.")

    # --- 8. Evaluasi Model (Opsional tapi penting) ---
    print("\n--- Hasil Evaluasi Model ---")
    y_pred_knn = knn_model.predict(X_test)
    print("\nLaporan Klasifikasi KNN:")
    print(classification_report(y_test, y_pred_knn, target_names=label_enc.classes_, zero_division=0))
    
    y_pred_nb = nb_model.predict(X_test)
    print("\nLaporan Klasifikasi Naive Bayes:")
    print(classification_report(y_test, y_pred_nb, target_names=label_enc.classes_, zero_division=0))
    print("---------------------------------")


    # --- 9. Simpan Model & Preprocessor ---
    print("\nMenyimpan model dan preprocessor ke file...")
    
    pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
    pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
    
    # Simpan semua preprocessor dalam satu file
    preprocessors = {
        'ordinal_encoder': ordinal_enc,
        'napza_vectorizer': napza_vec,
        'numerical_scaler': num_scaler,
        'label_encoder': label_enc
    }
    pickle.dump(preprocessors, open('preprocessors.pkl', 'wb'))
    
    print("\n--- SKRIP SELESAI ---")
    print("Berhasil membuat 3 file:")
    print("1. knn_model.pkl (Model KNN Anda)")
    print("2. nb_model.pkl (Model Naive Bayes Anda)")
    print("3. preprocessors.pkl (Pembersih data Anda)")