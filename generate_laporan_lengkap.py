import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- KONFIGURASI ---
output_folder = 'laporan_skripsi_final'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("=================================================")
print("   GENERATOR LAPORAN SKRIPSI (DENGAN URICA)      ")
print("=================================================")

# -------------------------------------------------------
# TAHAP 1: MUAT DATA & MODEL
# -------------------------------------------------------
print("\n[1] Memuat Model & Data...")

# A. Load Model
try:
    with open('model_rehab.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        knn = saved_data['knn_model']
        nb = saved_data['nb_model']
        le_gender = saved_data['le_gender']
        mlb_napza = saved_data['mlb_napza']
        mlb_penyakit = saved_data['mlb_penyakit']
        penyakit_classes = saved_data['penyakit_classes']
        # [BARU] Load Scaler URICA
        scaler_urica = saved_data['scaler_urica']
    print("    -> Model berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan train_model.py sudah dijalankan. {e}")
    exit()

# B. Load CSV
# Gunakan path file CSV terbaru Anda
csv_path = 'csv/DATA-PASIEN-REVISI-2.csv' 
try:
    # Coba load file revisi dulu
    data = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    print(f"    -> Data berhasil dimuat: {len(data)} baris.")
except:
    # Fallback jika file revisi belum ada di path itu, coba file lama atau lokal
    try:
        csv_path = 'DATA-PASIEN-REVISI-2.csv'
        data = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
        print(f"    -> Data berhasil dimuat (lokal): {len(data)} baris.")
    except Exception as e:
        print(f"ERROR: Gagal memuat CSV. {e}")
        exit()

# -------------------------------------------------------
# TAHAP 2: PRE-PROCESSING (Untuk Poin 1)
# -------------------------------------------------------
print("\n[2] Menghasilkan Output Poin 1: Preprocessing...")

# Ambil kolom index: 3(Gender), 5(Napza), 6(Lama), 7(Penyakit), 8(URICA), 9(Target)
# Sesuaikan index jika kolom CSV bergeser
try:
    df = data.iloc[:, [3, 5, 6, 7, 8, 9]].copy()
    df.columns = ['jenis_kelamin', 'jenis_napza', 'lama_penggunaan', 'riwayat_penyakit', 'urica_score', 'target']
except:
    # Fallback jika kolom kurang (misal file lama)
    print("Warning: Struktur kolom mungkin beda, mencoba mode kompatibilitas...")
    df = data.iloc[:, [3, 5, 6, 7, 8]].copy()
    df.columns = ['jenis_kelamin', 'jenis_napza', 'lama_penggunaan', 'riwayat_penyakit', 'target']
    df['urica_score'] = 0 # Default 0 jika kolom tidak ada

# Bersihkan Target
df['target'] = df['target'].astype(str).str.strip().str.title()
# Hapus Baris Kosong di Target
df = df.dropna(subset=['target']) 
counts = df['target'].value_counts()
to_remove = counts[counts < 2].index 
if len(to_remove) > 0:
    df = df[~df['target'].isin(to_remove)]

# A. URICA (Scaling Wajib)
df['urica_score'] = pd.to_numeric(df['urica_score'], errors='coerce').fillna(0)
urica_reshaped = df['urica_score'].values.reshape(-1, 1)
df['urica_encoded'] = scaler_urica.transform(urica_reshaped)

# B. Cleaning Lama Penggunaan (Tanpa Scaler)
def bersihkan_lama(x):
    x = str(x).lower().replace(',', '.')
    if 'tahun' in x or 'thn' in x:
        return float(''.join(c for c in x if c.isdigit() or c == '.'))
    elif 'bulan' in x or 'bln' in x:
        return float(''.join(c for c in x if c.isdigit() or c == '.')) / 12
    return 0.0
df['lama_encoded'] = df['lama_penggunaan'].apply(bersihkan_lama)

# C. Transformasi Fitur Lain
df['gender_encoded'] = df['jenis_kelamin'].apply(lambda x: 1 if str(x).strip() == 'Laki-Laki' else 0)

df['jenis_napza'] = df['jenis_napza'].fillna('Tidak Ada')
df['napza_list'] = df['jenis_napza'].apply(lambda x: [item.strip() for item in str(x).replace(';', ',').split(',')])
napza_encoded = mlb_napza.transform(df['napza_list'])
napza_df = pd.DataFrame(napza_encoded, columns=mlb_napza.classes_, index=df.index)

df['riwayat_penyakit'] = df['riwayat_penyakit'].fillna('Tidak Ada')
df['penyakit_list'] = df['riwayat_penyakit'].apply(lambda x: [item.strip() for item in str(x).replace(';', ',').split(',')])
penyakit_encoded = mlb_penyakit.transform(df['penyakit_list'])
penyakit_df = pd.DataFrame(penyakit_encoded, columns=penyakit_classes, index=df.index)

# Gabung Data (Sertakan URICA)
X = pd.concat([df[['gender_encoded', 'lama_encoded', 'urica_encoded']], napza_df, penyakit_df], axis=1)
y = df['target']

# --- OUTPUT POIN 1: TABEL PREPROCESSING ---
# Menyandingkan Data Asli vs Data Angka
sampel_gabung = pd.concat([df[['jenis_kelamin', 'lama_penggunaan', 'urica_score']].head(), 
                           X[['gender_encoded', 'lama_encoded', 'urica_encoded']].head()], axis=1)
sampel_gabung.to_csv(f'{output_folder}/1_tabel_preprocessing.csv')
print(f"    -> File '1_tabel_preprocessing.csv' dibuat.")

# -------------------------------------------------------
# TAHAP 3: KLASIFIKASI & DIAGRAM (Untuk Poin 2 & 5)
# -------------------------------------------------------
print("\n[3] Menghasilkan Output Poin 2 & 5: Diagram & Akurasi...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn) * 100
acc_nb = accuracy_score(y_test, y_pred_nb) * 100

# --- OUTPUT POIN 2: DIAGRAM PERBANDINGAN ---
plt.figure(figsize=(8, 6))
bars = plt.bar(['KNN', 'Naive Bayes'], [acc_knn, acc_nb], color=['#4F46E5', '#10B981'])
plt.title('Perbandingan Hasil Klasifikasi (Akurasi)', fontsize=14, fontweight='bold')
plt.ylabel('Akurasi (%)')
plt.ylim(0, 100)

# Menampilkan Label Akurasi (Poin 5)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.savefig(f'{output_folder}/2_grafik_perbandingan_klasifikasi.png')
plt.close()
print(f"    -> File '2_grafik_perbandingan_klasifikasi.png' dibuat.")

# -------------------------------------------------------
# TAHAP 4: EVALUASI (Untuk Poin 3)
# -------------------------------------------------------
print("\n[4] Menghasilkan Output Poin 3: Evaluasi (Confusion Matrix)...")

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(y_true.unique())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Evaluasi Model - {title}', fontsize=14)
    plt.xlabel('Prediksi Sistem')
    plt.ylabel('Data Aktual')
    plt.savefig(f'{output_folder}/{filename}')
    plt.close()

plot_cm(y_test, y_pred_knn, 'KNN', '3a_evaluasi_knn.png')
plot_cm(y_test, y_pred_nb, 'Naive Bayes', '3b_evaluasi_nb.png')
print(f"    -> File Evaluasi (3a & 3b) dibuat.")

# -------------------------------------------------------
# TAHAP 5: HASIL KLASIFIKASI LANJUTAN (Untuk Poin 4)
# -------------------------------------------------------
print("\n[5] Menghasilkan Output Poin 4: Hasil Klasifikasi Selanjutnya (Validitas)...")

def buat_grafik_validitas(y_true, y_pred, nama_algoritma, filename_img, filename_csv):
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    jumlah_asli = cm.sum(axis=1)      # Total Baris
    jumlah_valid = cm.diagonal()      # Diagonal (Benar)
    
    # Hitung Akurasi Total untuk Judul
    total_acc = (sum(jumlah_valid) / sum(jumlah_asli) * 100) if sum(jumlah_asli) > 0 else 0

    # Buat DataFrame
    df_val = pd.DataFrame({
        'Kategori': labels,
        'Data Asli': jumlah_asli,
        'Terklasifikasi Benar': jumlah_valid,
        'Persentase': [f"{(v/a*100):.1f}%" if a>0 else "0%" for v, a in zip(jumlah_valid, jumlah_asli)]
    })
    
    # Simpan CSV (Opsional)
    df_val.to_csv(f'{output_folder}/{filename_csv}', index=False, sep=';')

    # Visualisasi Grouped Bar Chart
    df_melt = df_val.melt(id_vars='Kategori', value_vars=['Data Asli', 'Terklasifikasi Benar'], 
                          var_name='Status', value_name='Jumlah')
    
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=df_melt, x='Kategori', y='Jumlah', hue='Status', palette=['#9CA3AF', '#10B981'])
    
    plt.title(f'Hasil Klasifikasi Per Kategori: {nama_algoritma}\n(Total Akurasi: {total_acc:.2f}%)', fontsize=14, fontweight='bold')
    plt.legend(title=None)
    plt.ylabel('Jumlah Pasien')
    
    # Label Data
    ax.bar_label(ax.containers[0], padding=3, fmt='%.0f') # Abu-abu
    
    # Label Hijau dengan Persen
    valid_labels = []
    for idx, row in df_val.iterrows():
        val = row['Terklasifikasi Benar']
        pct = row['Persentase']
        valid_labels.append(f"{val}\n({pct})")
    ax.bar_label(ax.containers[1], labels=valid_labels, padding=3, fontweight='bold', color='darkgreen')

    plt.tight_layout()
    plt.savefig(f'{output_folder}/{filename_img}')
    plt.close()

buat_grafik_validitas(y_test, y_pred_knn, 'KNN', '4a_hasil_klasifikasi_knn.png', '4a_tabel_knn.csv')
buat_grafik_validitas(y_test, y_pred_nb, 'Naive Bayes', '4b_hasil_klasifikasi_nb.png', '4b_tabel_nb.csv')
print(f"    -> File Hasil Klasifikasi (4a & 4b) dibuat.")

print("\n=================================================")
print(f" SELESAI! Semua file tersimpan di folder: '{output_folder}'")
print("=================================================")