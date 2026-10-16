import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

st.set_page_config(
    page_title="Dashboard Clustering Pendidikan",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Analisis Cluster Pendidikan SMA")
st.caption("Visualisasi hasil K-Means berdasarkan rasio sekolah, kepala sekolah, pendidik, dan peserta didik.")

uploaded = st.sidebar.file_uploader(
    "Unggah file Excel dataset",
    type=["xlsx", "xls", "csv"]
)

if uploaded is None:
    st.info("Unggah file Excel/CSV yang digunakan pada notebook Project_Skripsi.ipynb untuk memulai dashboard.")
    st.markdown("""
    **Dashboard ini mengikuti alur notebook:**
    1. Membaca dataset
    2. Membersihkan nilai kosong dan tak hingga
    3. Membentuk 4 variabel rasio
    4. Normalisasi dengan StandardScaler
    5. K-Means dengan K = 4
    6. Menampilkan evaluasi, grafik Elbow, jumlah anggota cluster, dan tabel hasil
    """)
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"File tidak dapat dibaca: {e}")
    st.stop()

required = ["Kota/Kab", "Sekolah", "Kepala Sekolah", "Pendidik", "Peserta Didik"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing)}")
    st.write("Kolom yang tersedia:", list(df.columns))
    st.stop()

initial_rows = len(df)
for c in ["Sekolah", "Kepala Sekolah", "Pendidik", "Peserta Didik"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required).copy()

# Hitung variabel rasio sesuai notebook pengguna
df["Rasio S/KS"] = df["Sekolah"] / df["Kepala Sekolah"].replace(0, np.nan)
df["Rasio S/P"] = df["Sekolah"] / df["Pendidik"].replace(0, np.nan)
df["Rasio P/PD"] = df["Pendidik"] / df["Peserta Didik"].replace(0, np.nan)
df["Persentase KS"] = (df["Kepala Sekolah"] / df["Sekolah"].replace(0, np.nan)) * 100

ratio_cols = ["Rasio S/KS", "Rasio S/P", "Rasio P/PD", "Persentase KS"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=ratio_cols).copy()

if len(df) < 5:
    st.error("Data valid terlalu sedikit untuk menjalankan K-Means dan evaluasi cluster.")
    st.stop()

X = df[ratio_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.sidebar.slider("Jumlah cluster (K)", min_value=2, max_value=min(10, len(df)-1), value=4)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

if len(set(df["Cluster"])) > 1 and len(df) > k:
    silhouette = silhouette_score(X_scaled, df["Cluster"])
    dbi = davies_bouldin_score(X_scaled, df["Cluster"])
    chi = calinski_harabasz_score(X_scaled, df["Cluster"])
else:
    silhouette, dbi, chi = np.nan, np.nan, np.nan


def beri_nama_cluster(frame, kolom, cluster_col="Cluster", ambang=0.10):
    """Beri nama karakteristik tiap cluster dari perbandingan dengan rata-rata umum."""
    peta = {}
    rata2_umum = frame[kolom].mean()
    for cid in sorted(frame[cluster_col].unique()):
        rata2_cluster = frame.loc[frame[cluster_col] == cid, kolom].mean()
        deviasi = (rata2_cluster - rata2_umum) / rata2_umum.abs().replace(0, np.nan)
        urut = deviasi.abs().sort_values(ascending=False)
        dipilih = [v for v in urut.index[:2] if pd.notna(deviasi[v]) and abs(deviasi[v]) >= ambang]
        if not dipilih:
            peta[cid] = "Mirip rata-rata umum"
        else:
            bagian = [f"{v} {'tinggi' if deviasi[v] > 0 else 'rendah'}" for v in dipilih]
            peta[cid] = ", ".join(bagian)
    return peta


cluster_ids = sorted(df["Cluster"].unique())
# Persentase KS = 100 / Rasio S/KS (informasi kembar), jadi tidak dipakai untuk penamaan
kolom_penamaan = ["Rasio S/KS", "Rasio S/P", "Rasio P/PD"]
nama_cluster = beri_nama_cluster(df, kolom_penamaan)
palette = px.colors.qualitative.Plotly
warna_cluster = {c: palette[i % len(palette)] for i, c in enumerate(cluster_ids)}
label_cluster = {c: f"C{c} · {nama_cluster[c]}" for c in cluster_ids}
warna_label = {label_cluster[c]: warna_cluster[c] for c in cluster_ids}
df["Karakter Cluster"] = df["Cluster"].map(label_cluster)

counts = df["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Jumlah Kabupaten/Kota")
counts["Karakter Cluster"] = counts["Cluster"].map(label_cluster)

# KPI
a, b, c, d = st.columns(4)
a.metric("Data valid", f"{len(df):,}")
b.metric("Jumlah cluster", str(k))
c.metric("Silhouette Score", f"{silhouette:.3f}" if pd.notna(silhouette) else "N/A")
d.metric("Davies-Bouldin Index", f"{dbi:.3f}" if pd.notna(dbi) else "N/A")

st.caption(f"Baris awal: {initial_rows:,} | Baris valid setelah pemeriksaan rasio: {len(df):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Ringkasan", "Elbow Method", "Karakteristik Cluster", "Data Hasil"
])

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Jumlah anggota setiap cluster")
        urut = counts.sort_values("Karakter Cluster", ascending=False)
        fig = px.bar(
            urut, x="Jumlah Kabupaten/Kota", y="Karakter Cluster", text="Jumlah Kabupaten/Kota",
            orientation="h", color="Karakter Cluster", color_discrete_map=warna_label,
            title="Distribusi Anggota Cluster"
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Jumlah kabupaten/kota")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Visualisasi cluster (nilai rasio asli)")
        plot_df = df[["Kota/Kab", "Rasio S/P", "Rasio P/PD", "Karakter Cluster"]].copy()
        plot_df["Kota/Kab"] = plot_df["Kota/Kab"].astype(str)

        # Deteksi sebaran miring (outlier): pakai skala logaritmik untuk tampilan
        skew_x = df["Rasio S/P"].skew()
        skew_y = df["Rasio P/PD"].skew()
        log_x, log_y = bool(skew_x > 1.5), bool(skew_y > 1.5)
        pakai_log = st.checkbox(
            "Skala logaritmik (meredam efek nilai ekstrem)",
            value=(log_x or log_y),
            help="Nilai data tidak diubah, hanya skala tampilan grafik agar titik tidak menumpuk. Nilai asli tetap terlihat saat kursor diarahkan ke titik."
        )
        tampilkan_label = st.checkbox(
            "Tampilkan nama kabupaten/kota pada grafik",
            value=len(plot_df) <= 50,
            help="Matikan jika titik terlalu padat atau saling menutupi."
        )
        fig = px.scatter(
            plot_df, x="Rasio S/P", y="Rasio P/PD",
            color="Karakter Cluster", color_discrete_map=warna_label,
            category_orders={"Karakter Cluster": [label_cluster[c] for c in cluster_ids]},
            hover_name="Kota/Kab",
            hover_data={"Rasio S/P": ":.2f", "Rasio P/PD": ":.2f", "Karakter Cluster": False},
            text="Kota/Kab" if tampilkan_label else None,
            title="Sebaran kabupaten/kota menurut rasio (satuan asli)",
            labels={
                "Rasio S/P": "Rasio Sekolah per Pendidik",
                "Rasio P/PD": "Rasio Pendidik per Peserta Didik",
            },
        )
        if tampilkan_label:
            fig.update_traces(textposition="top center", textfont=dict(size=9))
        if pakai_log:
            if not (log_x or log_y):
                log_x = log_y = True
            if log_x:
                fig.update_xaxes(type="log")
            if log_y:
                fig.update_yaxes(type="log")
        pusat = df.groupby("Cluster")[["Rasio S/P", "Rasio P/PD"]].mean()
        for cid in cluster_ids:
            fig.add_trace(go.Scatter(
                x=[pusat.loc[cid, "Rasio S/P"]],
                y=[pusat.loc[cid, "Rasio P/PD"]],
                mode="markers",
                name=f"★ Pusat C{cid}",
                marker=dict(symbol="star", size=17, color=warna_cluster[cid],
                            line=dict(width=1.5, color="#2b2b2b")),
                hovertemplate=(
                    f"Pusat C{cid} – {nama_cluster[cid]}<br>"
                    "Rasio S/P: %{x:.2f}<br>Rasio P/PD: %{y:.2f}<extra></extra>"
                ),
            ))
        st.plotly_chart(fig, use_container_width=True)
        if pakai_log and (log_x or log_y):
            sumbu_log = [s for s, f in [("Rasio S/P", log_x), ("Rasio P/PD", log_y)] if f]
            pesan_log = (
                f" Sumbu {' dan '.join(sumbu_log)} memakai skala logaritmik karena sebarannya miring "
                "(terdapat nilai ekstrem) sehingga titik tidak menumpuk di satu sudut. Nilai asli tetap tampil pada hover."
            )
        else:
            pesan_log = ""
        st.caption("Tanda ★ adalah pusat (rata-rata) tiap cluster." + pesan_log)
    st.metric("Calinski-Harabasz Index", f"{chi:.2f}" if pd.notna(chi) else "N/A")
    st.caption("SSE/Inertia mengukur jumlah kuadrat jarak data terhadap centroid; nilai lebih rendah umumnya menunjukkan cluster lebih rapat, tetapi perlu dibandingkan pada nilai K yang berbeda.")

with tab2:
    st.subheader("Elbow Method (WCSS)")
    max_k = min(10, len(df) - 1)
    ks = list(range(1, max_k + 1))
    wcss = []
    for n in ks:
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    elbow_df = pd.DataFrame({"K": ks, "WCSS": wcss})
    fig = px.line(elbow_df, x="K", y="WCSS", markers=True, title="Nilai WCSS untuk Setiap K")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(elbow_df, use_container_width=True)
    st.caption("Pilih K berdasarkan titik ketika penurunan WCSS mulai melandai. Slider K di sidebar mengubah jumlah cluster yang ditampilkan.")

with tab3:
    st.subheader("Karakteristik tiap cluster")
    tabel_karakter = counts.rename(columns={"Jumlah Kabupaten/Kota": "Jumlah anggota"})
    st.dataframe(
        tabel_karakter[["Cluster", "Karakter Cluster", "Jumlah anggota"]],
        use_container_width=True, hide_index=True
    )

    profile = df.groupby("Cluster")[ratio_cols].mean().round(4)
    profile_display = profile.copy()
    profile_display.index = [label_cluster[c] for c in profile_display.index]
    profile_display.index.name = "Cluster"

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Radar profil cluster")
        norm = profile.copy()
        for kolom in ratio_cols:
            selisih = norm[kolom].max() - norm[kolom].min()
            norm[kolom] = (norm[kolom] - norm[kolom].min()) / selisih if selisih > 0 else 0.5
        sumbu = ratio_cols + [ratio_cols[0]]
        fig = go.Figure()
        for cid in cluster_ids:
            nilai = norm.loc[cid, ratio_cols].tolist()
            nilai = nilai + [nilai[0]]
            fig.add_trace(go.Scatterpolar(
                r=nilai, theta=sumbu, fill="toself",
                name=label_cluster[cid],
                line=dict(color=warna_cluster[cid]),
                hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
            ))
        fig.update_layout(
            title="Bentuk karakter tiap cluster (skala relatif 0–1)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Nilai 0–1 menunjukkan posisi relatif rata-rata cluster dibandingkan cluster lain (0 = terendah, 1 = tertinggi).")
    with kanan:
        st.subheader("Profil rata-rata variabel rasio")
        profile_long = profile.reset_index().melt(id_vars="Cluster", var_name="Variabel", value_name="Rata-rata")
        profile_long["Karakter Cluster"] = profile_long["Cluster"].map(label_cluster)
        fig = px.bar(
            profile_long, x="Variabel", y="Rata-rata",
            color="Karakter Cluster", color_discrete_map=warna_label, barmode="group",
            category_orders={"Karakter Cluster": [label_cluster[c] for c in cluster_ids]},
            title="Profil Rata-rata Cluster (satuan asli)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabel rata-rata variabel rasio per cluster")
    st.dataframe(profile_display, use_container_width=True)

with tab4:
    st.subheader("Tabel hasil clustering")
    show_cols = ["Kota/Kab", "Sekolah", "Peserta Didik", "Kepala Sekolah", "Pendidik"] + ratio_cols + ["Cluster", "Karakter Cluster"]
    st.dataframe(df[show_cols].sort_values(["Cluster", "Kota/Kab"]), use_container_width=True)
    st.download_button(
        "⬇️ Unduh hasil clustering (CSV)",
        data=df[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_clustering_pendidikan.csv",
        mime="text/csv"
    )

with st.expander("Tentang metrik evaluasi"):
    st.write(f"**SSE / Inertia:** {model.inertia_:.4f}")
    st.write(f"**Silhouette Score:** {silhouette:.4f}" if pd.notna(silhouette) else "**Silhouette Score:** tidak dapat dihitung.")
    st.write(f"**Davies-Bouldin Index:** {dbi:.4f}" if pd.notna(dbi) else "**Davies-Bouldin Index:** tidak dapat dihitung.")
    st.write(f"**Calinski-Harabasz Index:** {chi:.4f}" if pd.notna(chi) else "**Calinski-Harabasz Index:** tidak dapat dihitung.")
    st.caption("Metrik dihitung ulang berdasarkan data yang diunggah dan nilai K yang dipilih. Hasil bisa berbeda jika dataset atau preprocessing berbeda.")
