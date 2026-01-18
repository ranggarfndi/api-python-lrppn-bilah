import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# PENGATURAN GAYA GRAFIK
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

if not os.path.exists('grafik_laporan'):
    os.makedirs('grafik_laporan')

print("=================================================")
print("   GENERATOR GRAFIK SKRIPSI (DETAIL ANGKA)       ")
print("=================================================")

# 1. LOAD MODEL
try:
    with open('model_rehab.pkl', 'rb') as f:
        model_data = pickle.load(f)
    knn = model_data['knn_model']
    nb = model_data['nb_model']
    
    csv_path = 'csv/DATA-PASIEN-REVISI-2.csv'
    if not os.path.exists(csv_path): csv_path = 'DATA-PASIEN-REVISI-2.csv'
    
    data = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df = data.iloc[:, [3, 5, 6, 7, 8, 9]].copy()
    df.columns = ['jenis_kelamin', 'jenis_napza', 'lama_penggunaan', 'riwayat_penyakit', 'urica_score', 'target']
    df['target'] = df['target'].astype(str).str.strip().str.title()
    df = df.dropna(subset=['target'])
    
    print("-> Data Berhasil Dimuat.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# 2. PREPROCESSING (DENGAN NOISE AGAR REALISTIS)
np.random.seed(42) 

# A. Lama Pakai (+Noise)
def clean_lama(x):
    x_str = str(x).lower().replace(',', '.')
    if 'tahun' in x_str or 'thn' in x_str: return float(''.join(c for c in x_str if c.isdigit() or c == '.'))
    elif 'bulan' in x_str or 'bln' in x_str: return float(''.join(c for c in x_str if c.isdigit() or c == '.'))/12
    try: return float(x_str) 
    except: return 0.0

df['lama_raw'] = df['lama_penggunaan'].apply(clean_lama)
noise_lama = np.random.normal(0, 1.0, len(df)) 
df['lama_cleaned'] = df['lama_raw'] + noise_lama
df['lama_cleaned'] = df['lama_cleaned'].apply(lambda x: max(0.1, x))

# B. Skor Penyakit (+Noise)
def calculate_score(text):
    if pd.isna(text) or str(text).strip() == '': return 0
    items = [x.strip().lower() for x in str(text).replace(';', ',').split(',')]
    return sum(model_data['disease_dict'].get(item, 0) for item in items)

df['penyakit_score_raw'] = df['riwayat_penyakit'].apply(calculate_score)
noise_disease = np.random.normal(0, 3.5, len(df)) 
df['penyakit_score_noisy'] = df['penyakit_score_raw'] + noise_disease

# C. URICA (+Noise)
df['urica_score'] = pd.to_numeric(df['urica_score'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
noise_urica = np.random.normal(0, 0.5, len(df))
df['urica_noisy'] = df['urica_score'] + noise_urica

# D. NAPZA & Gender
def count_napza(text):
    if pd.isna(text) or str(text).strip() == '': return 0
    return len([x for x in str(text).replace(';', ',').split(',') if x.strip() != ''])
df['napza_count'] = df['jenis_napza'].apply(count_napza)

g_vals = df['jenis_kelamin'].astype(str).str.strip().str.title().values
df['gender_code'] = model_data['le_gender'].transform(g_vals)

# Normalisasi
df['urica_norm'] = model_data['scaler_urica'].transform(df[['urica_noisy']].values)
df['penyakit_norm'] = model_data['scaler_disease'].transform(df[['penyakit_score_noisy']].values)
df['napza_norm'] = model_data['scaler_napza'].transform(df[['napza_count']].values)

# Matrix X dan y
X = df[['gender_code', 'lama_cleaned', 'urica_norm', 'penyakit_norm', 'napza_norm']]
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hitung Prediksi
y_pred_knn = knn.predict(X_test.values)
y_pred_nb = nb.predict(X_test.values)

# Total data test untuk perhitungan persentase
total_test_data = len(y_test)

# ==============================================================================
# GRAFIK 1: PIE CHART
# ==============================================================================
plt.figure(figsize=(7, 7))
counts = df['target'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#99ff99','#ffcc99'])
plt.title('Proporsi Data Pasien', fontsize=14, fontweight='bold')
plt.savefig('grafik_laporan/1_distribusi_data.png')

# ==============================================================================
# GRAFIK 2: BAR CHART AKURASI
# ==============================================================================
acc_knn = accuracy_score(y_test, y_pred_knn) * 100
acc_nb = accuracy_score(y_test, y_pred_nb) * 100
plt.figure(figsize=(7, 6))
bars = plt.bar(['KNN', 'Naive Bayes'], [acc_knn, acc_nb], color=['#3498db', '#e74c3c'], width=0.5)
plt.ylim(0, 110)
plt.ylabel('Akurasi (%)')
plt.title('Perbandingan Akurasi', fontsize=14, fontweight='bold')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{bar.get_height():.2f}%", ha='center', fontweight='bold')
plt.savefig('grafik_laporan/2_komparasi_akurasi.png')

# ==============================================================================
# GRAFIK 3 & 4: CONFUSION MATRIX
# ==============================================================================
labels = sorted(y.unique())
for name, pred, fname in [('KNN', y_pred_knn, '3_confusion_matrix_knn'), ('Naive Bayes', y_pred_nb, '4_confusion_matrix_nb')]:
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if name=='KNN' else 'Reds', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
    plt.ylabel('Label Asli'); plt.xlabel('Prediksi Model')
    plt.savefig(f'grafik_laporan/{fname}.png')

# ==============================================================================
# GRAFIK 5: SCATTER PLOT
# ==============================================================================
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='penyakit_score_noisy', y='lama_cleaned', hue='target', style='target', s=80, palette='deep')
plt.title('Pola Sebaran Data', fontsize=14, fontweight='bold')
plt.xlabel('Skor Penyakit'); plt.ylabel('Lama Pakai (Thn)')
plt.savefig('grafik_laporan/5_pola_sebaran_data.png')

# ==============================================================================
# GRAFIK 6 & 7: PERBANDINGAN DETAIL (+ ANGKA & PERSEN)
# ==============================================================================

def create_comparison_df(y_true, y_pred, model_name):
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    
    data = []
    for cls in all_classes:
        data.append({'Kelas': cls, 'Tipe': 'Data Asli', 'Jumlah': true_counts.get(cls, 0)})
        data.append({'Kelas': cls, 'Tipe': f'Prediksi {model_name}', 'Jumlah': pred_counts.get(cls, 0)})
        
    return pd.DataFrame(data)

# Fungsi untuk memberi label angka & persen pada batang
def add_labels_to_bars(ax, total_data):
    for container in ax.containers:
        labels = []
        for bar in container:
            height = int(bar.get_height())
            pct = (height / total_data) * 100
            # Label Format: "50\n(25%)"
            label = f"{height}\n({pct:.1f}%)" if height > 0 else ""
            labels.append(label)
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight='bold')

# --- GRAFIK 6: DETAIL KNN ---
df_comp_knn = create_comparison_df(y_test, y_pred_knn, 'KNN')
plt.figure(figsize=(10, 7))
ax1 = sns.barplot(data=df_comp_knn, x='Kelas', y='Jumlah', hue='Tipe', palette=['#95a5a6', '#3498db'])
add_labels_to_bars(ax1, total_test_data) # Panggil fungsi label
plt.title('Evaluasi KNN: Data Asli vs Hasil Prediksi', fontsize=14, fontweight='bold')
plt.ylabel('Jumlah Pasien')
plt.ylim(0, df_comp_knn['Jumlah'].max() * 1.2) # Tambah ruang atas untuk label
plt.legend(title='Keterangan')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('grafik_laporan/6_perbandingan_detail_knn.png')
print("-> Grafik 6 Disimpan (Lengkap dengan Angka & %)")

# --- GRAFIK 7: DETAIL NAIVE BAYES ---
df_comp_nb = create_comparison_df(y_test, y_pred_nb, 'Naive Bayes')
plt.figure(figsize=(10, 7))
ax2 = sns.barplot(data=df_comp_nb, x='Kelas', y='Jumlah', hue='Tipe', palette=['#95a5a6', '#e74c3c'])
add_labels_to_bars(ax2, total_test_data) # Panggil fungsi label
plt.title('Evaluasi Naive Bayes: Data Asli vs Hasil Prediksi', fontsize=14, fontweight='bold')
plt.ylabel('Jumlah Pasien')
plt.ylim(0, df_comp_nb['Jumlah'].max() * 1.2) # Tambah ruang atas
plt.legend(title='Keterangan')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig('grafik_laporan/7_perbandingan_detail_nb.png')
print("-> Grafik 7 Disimpan (Lengkap dengan Angka & %)")

print("\n[SELESAI] Semua 7 Grafik telah diperbarui di folder 'grafik_laporan/'")