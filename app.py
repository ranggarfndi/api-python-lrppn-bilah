import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import re
import sys
import warnings

# Abaikan peringatan untuk output yang bersih
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 1. Muat Model dan Preprocessor ---
print("Memuat model dan preprocessor...")
try:
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))
    nb_model = pickle.load(open('nb_model.pkl', 'rb'))
    preprocessors = pickle.load(open('preprocessors.pkl', 'rb'))
    
    # Ekstrak setiap preprocessor
    ordinal_enc = preprocessors['ordinal_encoder']
    napza_vec = preprocessors['napza_vectorizer']
    num_scaler = preprocessors['numerical_scaler']
    label_enc = preprocessors['label_encoder']
    
    print("Model dan preprocessor berhasil dimuat.")
except FileNotFoundError:
    print("Error: File .pkl tidak ditemukan. Pastikan 'train_model.py' sudah dijalankan.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error saat memuat file: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. Buat Aplikasi Flask ---
app = Flask(__name__)

# --- 3. Fungsi Helper: Pra-pemrosesan Input Mentah ---
def preprocess_input(raw_data):
    """
    Mengubah data JSON mentah dari Laravel menjadi 
    format array numpy yang dipahami model.
    """
    try:
        # Ekstrak data dari JSON
        jenis_kelamin = raw_data['jenis_kelamin']       # misal: "Laki-Laki"
        lama_penggunaan_raw = raw_data['lama_penggunaan'] # misal: "5 Tahun"
        jenis_napza_raw = raw_data['jenis_napza']         # misal: "Shabu; Ganja"
        
        # a) Proses 'Lama_Penggunaan'
        lama_penggunaan_clean = re.search(r'(\d+)', lama_penggunaan_raw)
        if not lama_penggunaan_clean:
            raise ValueError(f"Format 'Lama Penggunaan' tidak valid: {lama_penggunaan_raw}")
        lama_penggunaan_num = int(lama_penggunaan_clean.group(1))
        # Scale angka tersebut (harus dalam 2D array [[5]])
        lama_penggunaan_scaled = num_scaler.transform([[lama_penggunaan_num]])[0][0]
        
        # b) Proses 'Jenis Kelamin'
        try:
            jenis_kelamin_encoded = ordinal_enc.transform([[jenis_kelamin]])[0][0]
        except ValueError:
            raise ValueError(f"Format 'Jenis Kelamin' tidak valid: {jenis_kelamin}. Harap gunakan 'Laki-Laki' atau 'Perempuan'.")

        # c) Proses 'Jenis_NAPZA'
        jenis_napza_clean = re.sub(r'[;,]', ' ', jenis_napza_raw).lower()
        # Ubah teks napza menjadi array angka [0, 1, 0, 1, ...]
        napza_vector = napza_vec.transform([jenis_napza_clean]).toarray()[0]
        
        # d) Gabungkan semua fitur
        # Urutannya HARUS SAMA PERSIS seperti saat pelatihan
        base_features = [jenis_kelamin_encoded, lama_penggunaan_scaled]
        all_features = np.concatenate([base_features, napza_vector])
        
        # Kembalikan sebagai array 2D yang siap diprediksi
        return all_features.reshape(1, -1)
        
    except KeyError as e:
        print(f"Error: Input JSON kekurangan field: {e}", file=sys.stderr)
        raise ValueError(f"Input data kekurangan field: {e}")
    except Exception as e:
        print(f"Error saat pra-pemrosesan: {e}", file=sys.stderr)
        raise

# --- 4. Fungsi Helper: Rule Engine (Aturan Bisnis) ---
def get_recommendation(predicted_class_label):
    """
    Memetakan hasil prediksi ('Berat', 'Sedang', 'Ringan')
    ke rekomendasi program spesifik.
    (Aturan ini harus Anda konfirmasi dengan LRPPN BI)
    """
    if predicted_class_label == 'Berat':
        program = "Rawat Inap dan Rehabilitasi Medis"
        catatan = "Pasien membutuhkan detoksifikasi dan pengawasan medis penuh."
    elif predicted_class_label == 'Sedang':
        program = "Rawat Jalan dan Rehabilitasi Non-Medis (Sosial)"
        catatan = "Wajib lapor intensif dan konseling kelompok/individu."
    elif predicted_class_label == 'Ringan':
        program = "Rawat Jalan (Konseling Dasar)"
        catatan = "Pemantauan berkala dan konseling opsional."
    else:
        program = "Perlu Asesmen Manual"
        catatan = "Klasifikasi tidak teridentifikasi."
    return program, catatan

# --- 5. Definisikan Rute API Utama ---
@app.route("/predict", methods=['POST'])
def predict():
    """
    Endpoint utama yang akan dipanggil oleh Laravel.
    """
    print("Menerima permintaan di /predict ...")
    
    # a) Dapatkan data JSON dari request
    try:
        raw_data = request.get_json()
        if not raw_data:
            return jsonify({"error": "Tidak ada data input"}), 400
    except Exception as e:
        return jsonify({"error": f"Format JSON tidak valid: {e}"}), 400
        
    print(f"Data mentah diterima: {raw_data}")
    
    # b) Proses data mentah
    try:
        features = preprocess_input(raw_data)
    except Exception as e:
        # Jika pra-pemrosesan gagal (misal: format salah)
        return jsonify({"error": f"Gagal memproses input: {str(e)}"}), 400
    
    # c) Lakukan Prediksi
    try:
        # Dapatkan prediksi (sebagai angka, misal: [0])
        pred_knn_index = knn_model.predict(features)[0]
        pred_nb_index = nb_model.predict(features)[0]
        
        # Ubah angka prediksi kembali ke label (misal: "Berat")
        pred_knn_label = label_enc.inverse_transform([pred_knn_index])[0]
        pred_nb_label = label_enc.inverse_transform([pred_nb_index])[0]
        
    except Exception as e:
        print(f"Error saat prediksi: {e}", file=sys.stderr)
        return jsonify({"error": "Gagal melakukan prediksi. Cek log server."}), 500

    # d) Dapatkan Rekomendasi Program (berdasarkan hasil KNN)
    # Kita pilih KNN sebagai model utama untuk rekomendasi
    rekomendasi_program, catatan = get_recommendation(pred_knn_label)

    # e) Bangun Respon JSON untuk dikirim kembali ke Laravel
    response = {
        'input_data': raw_data,
        'prediksi_knn': {
            'tingkat_keparahan': pred_knn_label,
        },
        'prediksi_naive_bayes': {
            'tingkat_keparahan': pred_nb_label,
        },
        'rekomendasi_sistem': {
            'berdasarkan': f"Prediksi KNN ({pred_knn_label})",
            'program': rekomendasi_program,
            'catatan': catatan
        },
        'model_info': {
            'catatan': 'PERINGATAN: Akurasi model ini terbatas (sekitar 35-38%). Hasil ini harus digunakan sebagai PENDUKUNG KEPUTUSAN dan BUKAN pengganti asesmen ahli dari staf LRPPN BI.'
        }
    }
    
    print(f"Mengirim respon: {response}")
    return jsonify(response)

# --- 6. Jalankan Server ---
if __name__ == '__main__':
    # Host '0.0.0.0' membuat server ini bisa diakses 
    # oleh aplikasi lain (Laravel) di komputer Anda.
    # 'debug=True' akan otomatis me-restart server jika kita mengubah kode ini.
    print("Menjalankan Flask server di http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)