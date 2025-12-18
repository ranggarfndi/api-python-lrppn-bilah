import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # [BARU] Import Report

warnings.filterwarnings('ignore')

print("=================================================")
print("                 PELATIHAN AI                    ")
print("=================================================")

# --- 1. LOAD DATASET BARU ---
csv_path = 'csv/DATA-PASIEN-REVISI-2.csv'

if not os.path.exists(csv_path):
    print(f"ERROR: File {csv_path} tidak ditemukan.")
    # Fallback untuk testing lokal jika path beda
    csv_path = 'DATA-PASIEN-REVISI-2.csv'

try:
    print(f"[1] Membaca file: {csv_path}...")
    data = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    print(f"    -> Total Data: {len(data)} baris.")
except Exception as e:
    print(f"Error Membaca CSV: {e}")
    exit()

# --- 2. BERSIHKAN DATA ---
# [PENTING] Menggunakan Index Sesuai Permintaan Anda:
# Index 8 = URICA Score
# Index 9 = Target
try:
    df = data.iloc[:, [3, 5, 6, 7, 8, 9]].copy()
    df.columns = ['jenis_kelamin', 'jenis_napza', 'lama_penggunaan', 'riwayat_penyakit', 'urica_score', 'target']
except:
    print("ERROR: Posisi kolom tidak sesuai. Pastikan urutan kolom CSV benar.")
    exit()

# Bersihkan Target
df['target'] = df['target'].astype(str).str.strip().str.title()

# Hapus data kosong / sampah
df = df.dropna(subset=['target'])
counts = df['target'].value_counts()
to_remove = counts[counts < 2].index
if len(to_remove) > 0:
    df = df[~df['target'].isin(to_remove)]

print(f"    -> Data Bersih: {len(df)} baris.")

# --- 3. PRE-PROCESSING ---
print("\n[2] Preprocessing (URICA, Lama, Gender, dll)...")

# A. URICA SCORE (Wajib Scaler)
# Bersihkan dulu dari non-angka (jika ada)
df['urica_score'] = pd.to_numeric(df['urica_score'], errors='coerce').fillna(0)
scaler_urica = MinMaxScaler()
urica_reshaped = df['urica_score'].values.reshape(-1, 1)
df['urica_encoded'] = scaler_urica.fit_transform(urica_reshaped)

# B. Lama Penggunaan
def bersihkan_lama(x):
    x = str(x).lower().replace(',', '.')
    if 'tahun' in x or 'thn' in x:
        return float(''.join(c for c in x if c.isdigit() or c == '.'))
    elif 'bulan' in x or 'bln' in x:
        return float(''.join(c for c in x if c.isdigit() or c == '.')) / 12
    return 0.0
df['lama_encoded'] = df['lama_penggunaan'].apply(bersihkan_lama)

# C. Gender
le_gender = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['jenis_kelamin'].astype(str))

# D. NAPZA
df['jenis_napza'] = df['jenis_napza'].fillna('Tidak Ada')
df['napza_list'] = df['jenis_napza'].apply(lambda x: [item.strip() for item in str(x).replace(';', ',').split(',')])
mlb_napza = MultiLabelBinarizer()
napza_encoded = mlb_napza.fit_transform(df['napza_list'])
napza_df = pd.DataFrame(napza_encoded, columns=mlb_napza.classes_, index=df.index)

# E. Penyakit
df['riwayat_penyakit'] = df['riwayat_penyakit'].fillna('Tidak Ada')
df['penyakit_list'] = df['riwayat_penyakit'].apply(lambda x: [item.strip() for item in str(x).replace(';', ',').split(',')])
mlb_penyakit = MultiLabelBinarizer()
penyakit_encoded = mlb_penyakit.fit_transform(df['penyakit_list'])
penyakit_classes = ["SICK_" + str(c) for c in mlb_penyakit.classes_]
penyakit_df = pd.DataFrame(penyakit_encoded, columns=penyakit_classes, index=df.index)

# GABUNGKAN SEMUA
X = pd.concat([df[['gender_encoded', 'lama_encoded', 'urica_encoded']], napza_df, penyakit_df], axis=1)
y = df['target']

# --- 4. TRAINING & EVALUASI ---
print("\n[3] Mencari K Terbaik & Evaluasi Model...")

# Split Data (80% Train, 20% Test) untuk Evaluasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Cari K Terbaik
best_k = 5
best_score = 0
for k in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

print(f"    -> K Terbaik Ditemukan: {best_k} (Akurasi Validasi: {best_score*100:.2f}%)")

# --- [BARU] TAMPILKAN LAPORAN KLASIFIKASI ---
print("\n" + "="*20 + " LAPORAN KLASIFIKASI " + "="*20)

# Evaluasi KNN
knn_eval = KNeighborsClassifier(n_neighbors=best_k)
knn_eval.fit(X_train, y_train)
y_pred_knn = knn_eval.predict(X_test)
print("\n>> K-NEAREST NEIGHBORS (KNN):")
print(classification_report(y_test, y_pred_knn, zero_division=0))

# Evaluasi Naive Bayes
nb_eval = GaussianNB()
nb_eval.fit(X_train, y_train)
y_pred_nb = nb_eval.predict(X_test)
print("\n>> NAIVE BAYES (NB):")
print(classification_report(y_test, y_pred_nb, zero_division=0))

print("="*60)

# --- 5. TRAINING FINAL (FULL DATA) ---
print("\n[4] Melatih Model Final dengan Semua Data...")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X, y)

nb_final = GaussianNB()
nb_final.fit(X, y)

# Simpan Model & Scaler URICA
models_data = {
    'knn_model': knn_final,
    'nb_model': nb_final,
    'le_gender': le_gender,
    'mlb_napza': mlb_napza,
    'mlb_penyakit': mlb_penyakit,      
    'penyakit_classes': penyakit_classes,
    'scaler_urica': scaler_urica  # [PENTING]
}

with open('model_rehab.pkl', 'wb') as f:
    pickle.dump(models_data, f)

print(f"    -> SUKSES! Model disimpan sebagai 'model_rehab.pkl'.")
print("=================================================")