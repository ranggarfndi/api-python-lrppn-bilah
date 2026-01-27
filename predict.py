import sys
import json
import pickle
import numpy as np
import os
import warnings
import base64

warnings.filterwarnings("ignore")

# --- 0. CLASS PENERJEMAH NUMPY (SOLUSI ERROR INT64) ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 1. LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_rehab.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    
    knn = saved_data['knn_model']
    nb = saved_data['nb_model']
    le_gender = saved_data['le_gender']
    scaler_urica = saved_data['scaler_urica']
    scaler_disease = saved_data['scaler_disease']
    scaler_napza = saved_data['scaler_napza']
    DISEASE_SCORES = saved_data['disease_dict']

except Exception as e:
    # Gunakan NumpyEncoder di sini juga jaga-jaga errornya mengandung angka numpy
    print(json.dumps({'error': f"Gagal Load Model: {str(e)}"}, cls=NumpyEncoder))
    sys.exit(1)

# --- 2. FUNGSI PREPROCESSING ---
def clean_lama(x):
    x_str = str(x).lower().replace(',', '.')
    if 'tahun' in x_str or 'thn' in x_str: 
        return float(''.join(c for c in x_str if c.isdigit() or c == '.'))
    elif 'bulan' in x_str or 'bln' in x_str: 
        return float(''.join(c for c in x_str if c.isdigit() or c == '.'))/12
    try: return float(x_str)
    except: return 0.0

def calculate_score(text):
    if not text or str(text).strip().lower() in ['tidak ada', '', '-']: return 0
    items = [x.strip().lower() for x in str(text).replace(';', ',').split(',')]
    return sum(DISEASE_SCORES.get(item, 0) for item in items)

def count_napza(text):
    if not text or str(text).strip().lower() in ['tidak ada', '', '-']: return 0
    items = [x for x in str(text).replace(';', ',').split(',') if x.strip() != '']
    return len(items)

# --- 3. EKSEKUSI UTAMA ---
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(json.dumps({'error': 'Tidak ada data input'}))
            sys.exit(1)

        try:
            input_arg = sys.argv[1]
            input_json_str = base64.b64decode(input_arg).decode('utf-8')
            data = json.loads(input_json_str)
        except Exception as e:
            data = json.loads(sys.argv[1])

        # A. PREPROCESSING
        g_str = str(data.get('gender', 'Laki-Laki')).strip().title()
        val_gender = le_gender.transform([g_str])[0] if g_str in le_gender.classes_ else 0
        val_lama = clean_lama(data.get('lama_pakai', 0))
        
        try: raw_urica = float(data.get('urica', 0))
        except: raw_urica = 0.0
        val_urica = scaler_urica.transform([[raw_urica]])[0][0]

        val_sakit = scaler_disease.transform([[calculate_score(data.get('penyakit', ''))]])[0][0]
        val_napza = scaler_napza.transform([[count_napza(data.get('napza', ''))]])[0][0]

        X_input = np.array([[val_gender, val_lama, val_urica, val_sakit, val_napza]])

        # B. PREDIKSI
        
        # === KNN ===
        pred_knn = knn.predict(X_input)[0]
        try:
            knn_probs_arr = knn.predict_proba(X_input)[0]
            knn_conf = max(knn_probs_arr) * 100
        except: knn_conf = 0

        # [DETAIL JARAK TETANGGA]
        neighbors_dist, neighbors_idx = knn.kneighbors(X_input, n_neighbors=5)
        debug_knn = []
        for i in range(5):
            idx = neighbors_idx[0][i]
            dist = neighbors_dist[0][i]
            try:
                neighbor_label = knn._y[idx] 
            except:
                neighbor_label = "-"
            
            debug_knn.append({
                'rank': int(i+1),               # Pastikan int biasa
                'jarak': float(round(dist, 4)), # Pastikan float biasa
                'label': str(neighbor_label)    # Pastikan string biasa
            })

        # === NAIVE BAYES ===
        pred_nb = nb.predict(X_input)[0]
        try:
            nb_probs_arr = nb.predict_proba(X_input)[0]
            nb_conf = max(nb_probs_arr) * 100
            nb_detail_probs = {str(c): float(round(p*100, 2)) for c, p in zip(nb.classes_, nb_probs_arr)}
        except:
            nb_conf = 0
            nb_detail_probs = {}

        # C. OUTPUT JSON
        result = {
            'knn': {
                'label': str(pred_knn), # Paksa string
                'confidence': float(round(knn_conf, 1))
            },
            'nb': {
                'label': str(pred_nb),  # Paksa string
                'confidence': float(round(nb_conf, 1)),
                'probs': nb_detail_probs
            },
            'matrix_nilai': {
                'gender_num': int(val_gender),
                'lama_val': float(round(val_lama, 4)),
                'urica_norm': float(round(val_urica, 4)),
                'sakit_norm': float(round(val_sakit, 4)),
                'zat_norm': float(round(val_napza, 4))
            },
            'debug_knn': debug_knn
        }

        # [PENTING] Gunakan cls=NumpyEncoder agar int64 tidak error
        print(json.dumps(result, cls=NumpyEncoder))

    except Exception as e:
        print(json.dumps({'error': str(e)}, cls=NumpyEncoder))