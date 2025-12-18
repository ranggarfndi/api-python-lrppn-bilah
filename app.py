from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

print("=================================================")
print("             API REHABILITASI                    ")
print("=================================================")

# --- 1. LOAD MODEL ---
# Pastikan model_rehab.pkl sudah hasil training ulang dengan 1 kolom URICA
try:
    with open('model_rehab.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        knn = saved_data['knn_model']
        nb = saved_data['nb_model']
        le_gender = saved_data['le_gender']
        mlb_napza = saved_data['mlb_napza']
        mlb_penyakit = saved_data['mlb_penyakit']
        penyakit_classes = saved_data['penyakit_classes']
        # Pastikan scaler ini ada di file pickle (hasil training baru)
        scaler_urica = saved_data.get('scaler_urica', None) 
        
    print("    -> Model Berhasil Dimuat.")
    if scaler_urica is None:
        print("    [WARNING] 'scaler_urica' tidak ditemukan di model. Pastikan train_model.py sudah diperbarui.")

except Exception as e:
    print(f"    -> ERROR LOAD MODEL: {e}")
    print("    -> Susi: Pastikan Anda sudah menjalankan 'python train_model.py' yang baru.")
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
        
        # Input URICA Score (Float / Decimal)
        # Laravel mengirim key 'urica_score'
        input_urica = data.get('urica_score', 0) 

        # --- 3. PRE-PROCESSING ---
        
        # A. URICA (Scaling)
        # Mengubah angka skor (misal 3.5) menjadi skala yang dimengerti model (misal 0.5)
        try:
            val_urica = float(input_urica)
        except:
            val_urica = 0.0
            
        if scaler_urica:
            val_urica_scaled = scaler_urica.transform([[val_urica]])[0][0]
        else:
            # Fallback jika scaler lupa disimpan, pakai raw value (tidak disarankan untuk KNN)
            val_urica_scaled = val_urica

        # B. Lama Pakai (Parse String ke Float Tahun)
        lama_clean = str(input_lama).lower().replace(',', '.')
        val_lama = 0.0
        try:
            if 'tahun' in lama_clean or 'thn' in lama_clean:
                val_lama = float(''.join(c for c in lama_clean if c.isdigit() or c == '.'))
            elif 'bulan' in lama_clean or 'bln' in lama_clean:
                val_lama = float(''.join(c for c in lama_clean if c.isdigit() or c == '.')) / 12
            else:
                # Coba ambil angka saja jika user lupa nulis satuan
                val_lama = float(''.join(c for c in lama_clean if c.isdigit() or c == '.'))
        except:
            val_lama = 0.0
        
        # C. Gender (Label Encoding)
        try:
            val_gender = le_gender.transform([input_gender])[0]
        except:
            val_gender = 0 # Default jika error

        # D. NAPZA (One Hot / MultiLabel)
        list_napza = [item.strip() for item in str(input_napza).replace(';', ',').split(',')]
        vec_napza = mlb_napza.transform([list_napza])
        
        # E. Penyakit (One Hot / MultiLabel)
        list_penyakit = [item.strip() for item in str(input_penyakit).replace(';', ',').split(',')]
        vec_penyakit = mlb_penyakit.transform([list_penyakit])

        # --- 4. GABUNG DATA (DATAFRAME) ---
        # Urutan kolom di sini HARUS SAMA PERSIS dengan saat training (X.columns)
        
        # Kolom numerik utama
        df_main = pd.DataFrame([[val_gender, val_lama, val_urica_scaled]], 
                               columns=['gender_encoded', 'lama_encoded', 'urica_encoded'])
        
        # Kolom One-Hot NAPZA & Penyakit
        df_napza = pd.DataFrame(vec_napza, columns=mlb_napza.classes_)
        df_penyakit = pd.DataFrame(vec_penyakit, columns=penyakit_classes)
        
        # Gabung horizontal
        X_input = pd.concat([df_main, df_napza, df_penyakit], axis=1)

        # --- 5. PREDIKSI ---
        pred_knn = knn.predict(X_input)[0]
        pred_nb = nb.predict(X_input)[0]

        # --- 6. LOGIKA REKOMENDASI ---
        def get_program_info(label):
            if label in ['Berat', 'Sangat Berat']: 
                return 'Rawat Inap', 'Wajib detoksifikasi medis & pengawasan ketat 24 jam.'
            if label == 'Sedang': 
                return 'Rehabilitasi Rawat Jalan/Sosial', 'Fokus pada pemulihan perilaku dan konseling rutin.'
            # Ringan
            return 'Rawat Jalan (Konseling)', 'Konseling berkala dan edukasi pencegahan.'

        prog_knn, note_knn = get_program_info(pred_knn)
        prog_nb, note_nb = get_program_info(pred_nb)

        # --- 7. RESPONSE ---
        response = {
            'status': 'success',
            'prediksi_knn': {
                'tingkat_keparahan': pred_knn, 
                'program': prog_knn, 
                'catatan': note_knn
            },
            'prediksi_nb': {
                'tingkat_keparahan': pred_nb, 
                'program': prog_nb, 
                'catatan': note_nb
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"ERROR PREDICT: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)