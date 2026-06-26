# 🧠 Personality Predictor — Streamlit App

Aplikasi web untuk memprediksi tipe kepribadian (**Introvert** vs **Extrovert**)
berdasarkan kebiasaan sosial, menggunakan model **Random Forest Classifier**.

## 📦 Isi Folder

```
personality_app/
├── app.py                     # Aplikasi utama Streamlit
├── train_model.py             # Script untuk melatih ulang model (opsional)
├── personality_model.pkl      # Model Random Forest yang sudah dilatih
├── personality_dataset.csv    # Dataset
├── requirements.txt           # Daftar dependency Python
└── README.md                  # Dokumen ini
```

## 🚀 Cara Menjalankan di Lokal

1. **Buat virtual environment (opsional tapi disarankan):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

2. **Install dependency:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Opsional) Latih ulang model:**
   Model `personality_model.pkl` sudah disediakan. Jika ingin melatih ulang
   dari `personality_dataset.csv`:
   ```bash
   python train_model.py
   ```

4. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

5. Buka browser ke alamat yang muncul di terminal, biasanya:
   ```
   http://localhost:8501
   ```

## ☁️ Deployment ke Streamlit Community Cloud (Gratis)

1. Buat repository baru di GitHub, lalu upload semua file di folder ini
   (`app.py`, `train_model.py`, `personality_model.pkl`,
   `personality_dataset.csv`, `requirements.txt`).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **"New app"**, pilih repository tersebut.
4. Pada kolom **Main file path**, isi dengan `app.py`.
5. Klik **Deploy** dan tunggu beberapa menit hingga aplikasi aktif.

## ☁️ Deployment Alternatif (Lain)

- **Hugging Face Spaces**: pilih SDK "Streamlit", upload semua file di atas.
- **Railway / Render**: gunakan `requirements.txt` ini dan jalankan
  perintah start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## ✨ Fitur Aplikasi

- 🔮 **Halaman Prediksi** — Input data kebiasaan sosial via slider/radio
  interaktif, lalu dapatkan prediksi beserta tingkat keyakinan (gauge chart)
  dan breakdown probabilitas.
- 📈 **Halaman Eksplorasi Data** — Statistik dataset, distribusi target,
  feature importance, dan histogram interaktif per fitur.
- ℹ️ **Halaman Tentang Model** — Detail fitur, parameter model Random Forest,
  dan struktur project.
- 🎨 Desain modern dengan tema gradient ungu-pink, efek glassmorphism,
  dan animasi pada kartu hasil prediksi.

## 🧩 Catatan Teknis

- Kolom kategorikal `Stage_fear` dan `Drained_after_socializing` di-encode
  menjadi `No -> 0`, `Yes -> 1` menggunakan `LabelEncoder` yang disimpan
  bersama model di `personality_model.pkl`, sehingga mapping selalu konsisten
  saat inference.
- Target `Personality` di-encode `Extrovert -> 0`, `Introvert -> 1`, lalu
  di-decode kembali ke label aslinya saat menampilkan hasil ke pengguna.
- Model dilatih dengan parameter:
  `n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=1`.
