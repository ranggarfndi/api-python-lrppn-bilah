from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

print("=================================================")
print("             API REHABILITASI (UPDATED)          ")
print("=================================================")

# --- 1. LOAD MODEL TERBARU ---
# Memuat model yang menggunakan metode SCORING & NATURAL DISTRIBUTION
try:
    # Sesuaikan path jika file ada di folder tertentu (misal: 'python_engine/model_rehab.pkl')
    # Default kita asumsikan sejajar dengan app.py
    model_path = 'model_rehab.pkl' 
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        
        # Load Komponen Utama
        knn = saved_data['knn_model']
        nb = saved_data['nb_model']
        
        # Load Preprocessing Tools
        le_gender = saved_data['le_gender']
        scaler_urica = saved_data['scaler_urica']
        scaler_disease = saved_data['scaler_disease'] # Pengganti MLB Penyakit
        scaler_napza = saved_data['scaler_napza']     # Pengganti MLB NAPZA
        DISEASE_SCORES = saved_data['disease_dict']   # Kamus Skor Penyakit
        
    print("    -> Model Baru (Scoring Method) Berhasil Dimuat.")

except Exception as e:
    print(f"    -> ERROR LOAD MODEL: {e}")
    print("    -> Pastikan file 'model_rehab.pkl' hasil training terakhir sudah ada.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- 2. AMBIL INPUT ---
        input_gender = data.get('jenis_kelamin', 'Laki-Laki')
        input_lama = data.get('lama_penggunaan', '0 Tahun')
        input_napza = data.get('jenis_napza', 'Tidak Ada') 
        input_penyakit = data.get('riwayat_penyakit', 'Tidak Ada')
        input_urica = data.get('urica_score', 0) 

        # --- 3. PRE-PROCESSING (SESUAI TRAIN TERBARU) ---
        
        # A. Gender (Label Encoding)
        try:
            g_str = str(input_gender).strip().title()
            # Transform, handle error if unknown label
            if g_str in le_gender.classes_:
                val_gender = le_gender.transform([g_str])[0]
            else:
                val_gender = 0 # Default Laki-Laki
        except:
            val_gender = 0

        # B. Lama Pakai (Parsing String -> Float Tahun)
        def clean_lama(x):
            x_str = str(x).lower().replace(',', '.')
            if 'tahun' in x_str or 'thn' in x_str: 
                return float(''.join(c for c in x_str if c.isdigit() or c == '.'))
            elif 'bulan' in x_str or 'bln' in x_str: 
                return float(''.join(c for c in x_str if c.isdigit() or c == '.'))/12
            # Coba direct float
            try: return float(x_str)
            except: return 0.0

        val_lama = clean_lama(input_lama)

        # C. URICA (MinMax Scaling)
        try:
            raw_urica = float(input_urica)
        except:
            raw_urica = 0.0
        # Transform array 2D
        val_urica_norm = scaler_urica.transform([[raw_urica]])[0][0]

        # D. Penyakit (SCORING METHOD - BARU)
        # Menghitung total skor keparahan berdasarkan kamus
        def calculate_score(text):
            if not text or str(text).strip().lower() in ['tidak ada', '', '-']: return 0
            # Split koma, bersihkan spasi
            items = [x.strip().lower() for x in str(text).replace(';', ',').split(',')]
            # Sum skor dari dictionary
            total = sum(DISEASE_SCORES.get(item, 0) for item in items)
            return total

        raw_sakit_score = calculate_score(input_penyakit)
        val_sakit_norm = scaler_disease.transform([[raw_sakit_score]])[0][0]

        # E. NAPZA (COUNT METHOD - BARU)
        # Menghitung jumlah jenis zat
        def count_napza(text):
            if not text or str(text).strip().lower() in ['tidak ada', '', '-']: return 0
            items = [x for x in str(text).replace(';', ',').split(',') if x.strip() != '']
            return len(items)

        raw_napza_count = count_napza(input_napza)
        val_napza_norm = scaler_napza.transform([[raw_napza_count]])[0][0]

        # --- 4. SUSUN INPUT MATRIX ---
        # Format harus sama persis dengan training: 
        # [Gender, Lama, URICA_Norm, Sakit_Norm, Zat_Norm]
        X_input = np.array([[val_gender, val_lama, val_urica_norm, val_sakit_norm, val_napza_norm]])

        # --- 5. PREDIKSI ---
        pred_knn = knn.predict(X_input)[0]
        pred_nb = nb.predict(X_input)[0]

        # (Opsional) Ambil Probabilitas untuk Naive Bayes
        try:
            nb_probs = nb.predict_proba(X_input)[0]
            # Mapping kelas ke probabilitas
            probs_dict = {str(c): round(p * 100, 2) for c, p in zip(nb.classes_, nb_probs)}
        except:
            probs_dict = {}

        # --- 6. LOGIKA REKOMENDASI (FITUR LAMA TETAP ADA) ---
        def get_program_info(label):
            label = str(label).title()
            if label in ['Berat', 'Sangat Berat']: 
                return 'Rawat Inap', 'Wajib detoksifikasi medis & pengawasan ketat 24 jam.'
            if label == 'Sedang': 
                return 'Rehabilitasi Rawat Jalan/Sosial', 'Fokus pada pemulihan perilaku dan konseling rutin.'
            # Ringan
            return 'Rawat Jalan (Konseling)', 'Konseling berkala dan edukasi pencegahan.'

        prog_knn, note_knn = get_program_info(pred_knn)
        prog_nb, note_nb = get_program_info(pred_nb)

        # --- 7. RESPONSE JSON ---
        response = {
            'status': 'success',
            'debug_info': {
                'lama_pakai_tahun': val_lama,
                'urica_input': raw_urica,
                'total_skor_penyakit': raw_sakit_score,
                'jumlah_zat_napza': raw_napza_count
            },
            'prediksi_knn': {
                'tingkat_keparahan': pred_knn, 
                'program': prog_knn, 
                'catatan': note_knn
            },
            'prediksi_nb': {
                'tingkat_keparahan': pred_nb, 
                'program': prog_nb, 
                'catatan': note_nb,
                'detail_probabilitas': probs_dict # Tambahan fitur baru
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"ERROR PREDICT: {e}")
        import traceback
        traceback.print_exc() # Print error lengkap di terminal untuk debugging
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Gunakan host 0.0.0.0 agar bisa diakses dari luar localhost jika perlu
    app.run(debug=True, host='0.0.0.0', port=5000)