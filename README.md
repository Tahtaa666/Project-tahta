# Dashboard Analisis Cluster Pendidikan SMA

Dashboard interaktif berbasis **Streamlit** untuk memvisualisasikan hasil **K-Means Clustering** pada data pendidikan SMA (rasio sekolah, kepala sekolah, pendidik, dan peserta didik per kota/kabupaten).

Dashboard ini melengkapi analisis pada notebook [`Project_Skripsi.ipynb`](Project_Skripsi.ipynb).

## Fitur

- Unggah dataset sendiri (Excel/CSV) melalui sidebar
- Perhitungan otomatis 4 variabel rasio: Rasio S/KS, Rasio S/P, Rasio P/PD, dan Persentase KS
- Normalisasi data dengan `StandardScaler`, lalu K-Means clustering
- Slider jumlah cluster (K) di sidebar
- Evaluasi cluster: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, dan SSE/Inertia
- Grafik Elbow Method (WCSS)
- Visualisasi sebaran cluster (scatter + pusat cluster), radar karakteristik, dan profil rata-rata per cluster
- Tabel hasil clustering + tombol unduh hasil dalam format CSV

## Struktur file

| File | Keterangan |
|---|---|
| `app_dashboard_skripsi.py` | Aplikasi dashboard Streamlit |
| `requirements.txt` | Daftar dependensi Python |
| `Project_Skripsi.ipynb` | Notebook analisis (pipeline asli) |

## Cara menjalankan

1. (Opsional) Buat virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan dashboard:

   ```bash
   streamlit run app_dashboard_skripsi.py
   ```

4. Buka browser di `http://localhost:8501`, lalu unggah file Excel/CSV dataset melalui sidebar.

## Format dataset

File Excel/CSV harus memiliki kolom minimal berikut:

- `Kota/Kab`
- `Sekolah`
- `Kepala Sekolah`
- `Pendidik`
- `Peserta Didik`
