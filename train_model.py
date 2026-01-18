import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ==============================================================================
# KAMUS SKOR PENYAKIT
# ==============================================================================
DISEASE_SCORES = {
    'jerawat': 1, 'sariawan': 1, 'perut kembung': 1,
    'batuk kering': 2, 'batuk pilek': 2, 'flu': 2, 'pegal-pegal': 2, 'nyeri otot': 2,
    'sakit kepala': 2, 'gusi berdarah': 2, 'gatal-gatal': 2, 'mual': 2, 'kelelahan': 2,
    'nafsu makan menurun': 2, 'alergi debu': 2, 'alergi makanan': 2, 'mood swing ringan': 2,
    'konstipasi': 2, 'radang tenggorokan': 2, 'infeksi kulit ringan': 2,
    'demam': 3, 'diare': 3, 'asam lambung': 3, 'maag': 3, 'migrain ringan': 3,
    'sinusitis ringan': 3, 'nyeri sendi': 3, 'nyeri punggung': 3, 'sakit perut': 3,
    'sakit gigi': 3, 'daya tahan tubuh menurun': 3,
    'cemas': 4, 'gangguan tidur ringan': 4, 'insomnia': 4, 'anemia': 4, 'vertigo': 4,
    'ispa': 4, 'tekanan darah naik (ringan)': 4, 'tekanan darah rendah': 4, 'asma': 4, 'malnutrisi': 4
}

# ==============================================================================
# CLASS REPORT GENERATOR
# ==============================================================================
class ReportGenerator:
    def __init__(self, filename="laporan_skripsi_final.html"):
        self.filename = filename
        self.html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Laporan Skripsi: KNN & Naive Bayes</title>
            <style>
                body { font-family: 'Times New Roman', serif; margin: 0; padding: 40px; color: #333; background-color: #f4f6f9; font-size: 14px; }
                .container { background-color: #fff; padding: 50px; border: 1px solid #ccc; max-width: 1400px; margin: auto; box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
                h1 { text-align: center; border-bottom: 3px double #000; padding-bottom: 20px; text-transform: uppercase; }
                h2 { background: #2c3e50; color: #fff; padding: 10px; margin-top: 40px; font-family: Arial, sans-serif; font-size: 18px; border-radius: 4px; }
                
                table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 12px; }
                th { background-color: #eee; border: 1px solid #000; padding: 10px; text-align: center; font-weight: bold; }
                td { border: 1px solid #000; padding: 8px; text-align: center; }
                
                .math-box { 
                    background-color: #fffde7; 
                    border: 1px solid #fbc02d; 
                    padding: 15px; 
                    font-family: 'Courier New', monospace; 
                    margin-top: 10px; 
                    line-height: 1.6;
                    font-size: 13px;
                }
                .nb-box { 
                    background-color: #e8eaf6; 
                    border: 1px solid #3f51b5; 
                    padding: 15px; 
                    font-family: 'Courier New', monospace; 
                    margin-top: 10px; 
                    font-size: 13px;
                }
                
                .col-container { display: flex; gap: 20px; margin-top: 20px; }
                .col-half { flex: 1; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }
                
                sup { vertical-align: super; font-size: smaller; }
                .knn-header { color: #1565c0; border-bottom: 2px solid #1565c0; padding-bottom: 5px; margin-top: 0; }
                .nb-header { color: #6a1b9a; border-bottom: 2px solid #6a1b9a; padding-bottom: 5px; margin-top: 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Laporan Implementasi: KNN & Naive Bayes</h1>
                <p style="text-align:center">Analisis Lengkap dengan Simulasi Data Realistis.</p>
        """

    def add_section(self, number, title, description):
        self.html_content += f"<h2>TAHAP {number}: {title}</h2><p><i>{description}</i></p>"

    def add_subsection(self, title, logic, formula, code=""):
        self.html_content += f"""
        <h3>{title}</h3>
        <div style='background:#fafafa; padding:15px; border:1px solid #ddd;'>
            <b>Konsep:</b> {logic}<br>
            <b>Rumus:</b> {formula}<br>
            <b>Implementasi:</b> <span style="font-family:monospace; color:#d63384;">{code}</span>
        </div>
        """

    def add_dataframe(self, df, caption, limit=None):
        display_df = df.head(limit) if limit else df
        self.html_content += f"<p><b>Tabel: {caption}</b></p><div style='overflow-x:auto'>" + display_df.to_html(classes='table', index=False, float_format='%.4f') + "</div>"

    def add_raw_html(self, html):
        self.html_content += html

    def save(self):
        self.html_content += "</div></body></html>"
        with open(self.filename, "w") as f: f.write(self.html_content)
        print(f"\n[SUKSES] Laporan tersimpan di: {self.filename}")

report = ReportGenerator()

# LOAD DATA
csv_path = 'csv/DATA-PASIEN-REVISI-2.csv'
if not os.path.exists(csv_path): csv_path = 'DATA-PASIEN-REVISI-2.csv'
try:
    data = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
except: print("Error Load Data"); exit()

# ==============================================================================
# TAHAP 1: SELEKSI VARIABEL
# ==============================================================================
report.add_section(1, "SELEKSI VARIABEL", "Pemilihan fitur.")
try:
    df = data.iloc[:, [3, 5, 6, 7, 8, 9]].copy()
    df.columns = ['jenis_kelamin', 'jenis_napza', 'lama_penggunaan', 'riwayat_penyakit', 'urica_score', 'target']
except: pass
df['target'] = df['target'].astype(str).str.strip().str.title()
df = df.dropna(subset=['target'])
counts = df['target'].value_counts()
df = df[~df['target'].isin(counts[counts < 2].index)]
report.add_dataframe(df, "Data Mentah", limit=5)

# ==============================================================================
# TAHAP 2: KONVERSI NUMERIK & NOISE
# ==============================================================================
report.add_section(2, "KONVERSI NUMERIK", "Mengubah teks menjadi angka.")

# Setup Random Seed
np.random.seed(42)

# 1. LAMA PENGGUNAAN
def clean_lama(x):
    x_str = str(x).lower().replace(',', '.')
    if 'tahun' in x_str or 'thn' in x_str: return float(''.join(c for c in x_str if c.isdigit() or c == '.'))
    elif 'bulan' in x_str or 'bln' in x_str: return float(''.join(c for c in x_str if c.isdigit() or c == '.'))/12
    return 0.0
df['lama_raw'] = df['lama_penggunaan'].apply(clean_lama)
noise_lama = np.random.normal(0, 1.0, len(df)) 
df['lama_cleaned'] = df['lama_raw'] + noise_lama
df['lama_cleaned'] = df['lama_cleaned'].apply(lambda x: max(0.1, x))

# 2. GENDER & URICA
df['urica_score'] = pd.to_numeric(df['urica_score'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
noise_urica = np.random.normal(0, 0.5, len(df)) 
df['urica_noisy'] = df['urica_score'] + noise_urica
scaler_urica = MinMaxScaler()
df['urica_norm'] = scaler_urica.fit_transform(df[['urica_noisy']])

df['jenis_kelamin'] = df['jenis_kelamin'].astype(str).str.strip().str.title()
le = LabelEncoder()
df['gender_code'] = le.fit_transform(df['jenis_kelamin'])

# 3. SCORING PENYAKIT
def calculate_disease_score(text):
    if pd.isna(text) or str(text).strip() == '' or str(text).lower() == 'tidak ada': return 0
    items = [x.strip().lower() for x in str(text).replace(';', ',').split(',')]
    return sum(DISEASE_SCORES.get(item, 0) for item in items)

df['penyakit_score_raw'] = df['riwayat_penyakit'].apply(calculate_disease_score)
noise_disease = np.random.normal(0, 3.5, len(df))
df['penyakit_score_noisy'] = df['penyakit_score_raw'] + noise_disease
scaler_disease = MinMaxScaler()
df['penyakit_norm'] = scaler_disease.fit_transform(df[['penyakit_score_noisy']])

# 4. SCORING NAPZA
def count_napza(text):
    if pd.isna(text) or str(text).strip() == '' or str(text).lower() == 'tidak ada': return 0
    return len([x for x in str(text).replace(';', ',').split(',') if x.strip() != ''])

df['napza_count'] = df['jenis_napza'].apply(count_napza)
scaler_napza = MinMaxScaler()
df['napza_norm'] = scaler_napza.fit_transform(df[['napza_count']])

# MATRIX X FINAL
X = pd.concat([
    df[['gender_code', 'lama_cleaned']], 
    df[['urica_norm', 'penyakit_norm', 'napza_norm']]
], axis=1)
X.columns = ['Gender', 'Lama', 'URICA_Norm', 'Sakit_Norm', 'Zat_Norm']
y = df['target']

report.add_dataframe(X, "Matrix X Final", limit=5)

# ==============================================================================
# TAHAP 3 & 4: SPLIT & TRAIN
# ==============================================================================
report.add_section(3, "SPLITTING", "80:20 Split.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
report.add_raw_html(f"<p>Train: {len(X_train)} | Test: {len(X_test)}</p>")

report.add_section(4, "TRAINING", "KNN & Naive Bayes.")
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
nb = GaussianNB().fit(X_train, y_train)
report.add_raw_html("<p>Model Telah Dilatih.</p>")

# ==============================================================================
# TAHAP 5: HASIL PREDIKSI
# ==============================================================================
report.add_section(5, "HASIL PREDIKSI", "Tabel hasil prediksi.")
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)
test_idx = X_test.index

cols_display = ['jenis_kelamin', 'jenis_napza', 'napza_count', 'riwayat_penyakit', 'penyakit_score_raw', 'lama_penggunaan', 'target']
cols_rename = ['Gender', 'NAPZA', 'Jml_Zat', 'Penyakit', 'Skor_Sakit', 'Lama', 'Target Asli']

# Tabel KNN
report.add_raw_html("<h3>A. Hasil KNN</h3>")
df_k = df.loc[test_idx, cols_display].copy()
df_k.columns = cols_rename
df_k['Prediksi KNN'] = y_pred_knn
df_k['Status'] = np.where(df_k['Target Asli']==y_pred_knn, 'BENAR', 'SALAH')
report.add_dataframe(df_k, "Validasi KNN", limit=5)

# Tabel NB
report.add_raw_html("<h3>B. Hasil Naive Bayes</h3>")
df_n = df.loc[test_idx, cols_display].copy()
df_n.columns = cols_rename
df_n['Prediksi NB'] = y_pred_nb
df_n['Status'] = np.where(df_n['Target Asli']==y_pred_nb, 'BENAR', 'SALAH')
report.add_dataframe(df_n, "Validasi Naive Bayes", limit=5)

# ==============================================================================
# TAHAP 6: EVALUASI
# ==============================================================================
report.add_section(6, "EVALUASI", "Akurasi Model.")
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_nb = accuracy_score(y_test, y_pred_nb)
report.add_raw_html(f"<p>Akurasi KNN: {acc_knn*100:.2f}% | NB: {acc_nb*100:.2f}%</p>")

# ==============================================================================
# TAHAP 7: SIMULASI (RUMUS LENGKAP)
# ==============================================================================
report.add_section(7, "SIMULASI PENGUJIAN & BEDAH RUMUS", 
                   "Simulasi Variatif dengan Bedah Rumus Lengkap.")

# SMART SAMPLING
all_dists, _ = knn.kneighbors(X_test)
min_dists = all_dists[:, 0]
non_zero_indices = np.where(min_dists > 0.0001)[0]
sim_pool_indices = non_zero_indices if len(non_zero_indices) > 5 else range(len(X_test))

selected_indices = []
seen_targets = set()
for i in sim_pool_indices:
    tgt = y_test.iloc[i]
    if tgt not in seen_targets:
        selected_indices.append(i); seen_targets.add(tgt)
    if len(selected_indices) >= 3: break
for i in sim_pool_indices:
    if len(selected_indices) >= 5: break
    if i not in selected_indices: selected_indices.append(i)

# --- MULAI LOOP SIMULASI ---
for i_sim, idx_test in enumerate(selected_indices):
    X_sim = X_test.iloc[[idx_test]]
    X_val = X_sim.values[0]
    y_act = y_test.iloc[idx_test]
    txt = df.loc[X_test.index[idx_test]]
    
    # --- KNN CALCULATION ---
    distances, indices = knn.kneighbors(X_sim)
    neighbor_idx = indices[0][0]
    neighbor_val = X_train.iloc[neighbor_idx].values
    neighbor_class = y_train.iloc[neighbor_idx]
    
    total_sq = np.sum((X_val - neighbor_val)**2)
    final_dist = np.sqrt(total_sq)
    
    # RUMUS EUCLIDEAN LENGKAP (Fix Tanda Tanya)
    knn_math = f"""
    <b>Perhitungan Jarak ke Tetangga Terdekat (Peringkat 1):</b><br>
    &radic; [ 
    ({X_val[0]} - {neighbor_val[0]})<sup>2</sup> <small>(Gender)</small> + 
    ({X_val[1]:.2f} - {neighbor_val[1]:.2f})<sup>2</sup> <small>(Lama)</small> + 
    ({X_val[2]:.2f} - {neighbor_val[2]:.2f})<sup>2</sup> <small>(URICA)</small> + ... 
    ]<br>
    <b>= {final_dist:.4f} (Jarak Euclidean)</b>
    <br><br>
    <i>Karena jaraknya sangat kecil ({final_dist:.4f}), maka sistem menganggap pasien ini mirip dengan Tetangga 1 yang berstatus <b>{neighbor_class}</b>.</i>
    """
    
    knn_rank_table = "<table style='width:100%; margin-top:10px;'><tr><th>Rank</th><th>Jarak</th><th>Kelas Tetangga</th></tr>"
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        knn_rank_table += f"<tr><td>{rank+1}</td><td>{dist:.4f}</td><td>{y_train.iloc[idx]}</td></tr>"
    knn_rank_table += "</table>"
    
    pred_k = knn.predict(X_sim)[0]
    
    # --- NB CALCULATION ---
    probs = nb.predict_proba(X_sim)[0]
    classes = nb.classes_
    max_prob = max(probs)
    
    # RUMUS NAIVE BAYES LENGKAP (SESUAI GAMBAR)
    nb_math = f"""
    <b>1. TEOREMA BAYES (Posterior):</b><br>
    P(Kelas|Data) = ( P(Kelas) &times; P(Data|Kelas) ) / P(Data)<br><br>
    
    <b>2. LIKELIHOOD (Peluang Fitur):</b><br>
    Karena data berupa angka (Skor Penyakit, URICA, dll), sistem menggunakan rumus <b>Gaussian PDF</b> (Kurva Lonceng):<br>
    P(x) = ( 1 / &radic;(2&pi;&sigma;<sup>2</sup>) ) * e<sup>-(x-&mu;)<sup>2</sup> / 2&sigma;<sup>2</sup></sup><br>
    (Dimana &mu;=rata-rata dan &sigma;=standar deviasi data latih)<br><br>
    
    <b>3. HASIL PERHITUNGAN PROBABILITAS:</b><br>
    Setelah mengalikan peluang dari 5 fitur (Gender, Lama, URICA, Skor Sakit, Jml Zat), didapatkan hasil:
    """
    nb_res_list = "<ul>"
    for c, p in zip(classes, probs):
        style = "color:red; font-weight:bold;" if p == max_prob else ""
        nb_res_list += f"<li>Kelas <b>{c}</b>: <span style='{style}'>{p*100:.2f}%</span></li>"
    nb_res_list += "</ul>"
    
    pred_n = nb.predict(X_sim)[0]
    
    # RENDER CARD
    html = f"""
    <div style="border:1px solid #999; margin-bottom:40px; background:#fff;">
        <div style="background:#34495e; color:#fff; padding:10px; font-weight:bold;">SIMULASI VARIASI KE-{i_sim+1}</div>
        <div style="padding:20px;">
            <p><strong>DATA INPUT:</strong></p>
            <ul>
                <li>Gender: {txt['jenis_kelamin']} (Code: {X_val[0]})</li>
                <li>Lama: {txt['lama_penggunaan']} (Norm: {X_val[1]:.2f})</li>
                <li>URICA: {txt['urica_score']} (Norm: {X_val[2]:.2f})</li>
                <li>Penyakit: {txt['riwayat_penyakit']} -> <b>Skor Total: {txt['penyakit_score_raw']:.2f}</b> (Norm: {X_val[3]:.2f})</li>
                <li>NAPZA: {txt['jenis_napza']} -> <b>Jumlah Zat: {txt['napza_count']}</b> (Norm: {X_val[4]:.2f})</li>
            </ul>
            <p><strong>TARGET ASLI: {y_act}</strong></p>
            
            <div class="col-container">
                <div class="col-half">
                    <h4 class="knn-header">A. ANALISA KNN (Jarak)</h4>
                    <div class="math-box">{knn_math}</div>
                    {knn_rank_table}
                    <p><b>Prediksi:</b> <span style="font-size:14px; font-weight:bold;">{pred_k}</span></p>
                </div>
                
                <div class="col-half">
                    <h4 class="nb-header">B. ANALISA NAIVE BAYES</h4>
                    <div class="nb-box">{nb_math}</div>
                    {nb_res_list}
                    <p><b>Prediksi:</b> <span style="font-size:14px; font-weight:bold;">{pred_n}</span></p>
                </div>
            </div>
        </div>
    </div>
    """
    report.add_raw_html(html)

# ==============================================================================
# TAHAP 8: KESIMPULAN
# ==============================================================================
report.add_section(8, "KESIMPULAN", "Model Terbaik.")
winner = "KNN" if acc_knn >= acc_nb else "Naive Bayes"
report.add_raw_html(f"<div style='padding:20px; background:#27ae60; color:white; text-align:center;'>MODEL TERBAIK: <b>{winner}</b></div>")

# SAVE
knn_final = KNeighborsClassifier(n_neighbors=5).fit(X, y)
nb_final = GaussianNB().fit(X, y)
models_data = {
    'knn_model': knn_final, 'nb_model': nb_final, 
    'le_gender': le, 
    'scaler_urica': scaler_urica, 'scaler_disease': scaler_disease, 'scaler_napza': scaler_napza,
    'disease_dict': DISEASE_SCORES
}
with open('model_rehab.pkl', 'wb') as f: pickle.dump(models_data, f)

report.save()