import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS — TAMPILAN MENARIK
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(120deg, #7F00FF 0%, #E100FF 100%);
        padding: 2.2rem 2rem;
        border-radius: 22px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(127, 0, 255, 0.35);
        text-align: center;
    }
    .hero-title {
        color: white;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Card style */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .glass-card h3 {
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .section-label {
        color: #C9A8FF;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* Result cards */
    .result-extrovert {
        background: linear-gradient(135deg, #FF8008 0%, #FFC837 100%);
        border-radius: 22px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 35px rgba(255, 128, 8, 0.4);
        animation: popIn 0.5s ease-out;
    }
    .result-introvert {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        border-radius: 22px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 35px rgba(33, 147, 176, 0.4);
        animation: popIn 0.5s ease-out;
    }
    @keyframes popIn {
        0% { transform: scale(0.85); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.2rem;
    }
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .result-desc {
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric pills */
    .metric-pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-pill .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-pill .label {
        font-size: 0.8rem;
        color: #C9A8FF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }

    /* Streamlit widgets text color fix */
    .stMarkdown, .stMarkdown p, label, .stSlider label, .stRadio label {
        color: #e8e8f5 !important;
    }

    h1, h2, h3, h4 { color: #ffffff; }

    /* Button */
    .stButton > button {
        background: linear-gradient(120deg, #7F00FF 0%, #E100FF 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 2rem;
        border-radius: 14px;
        border: none;
        box-shadow: 0 6px 20px rgba(127, 0, 255, 0.4);
        transition: transform 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(127, 0, 255, 0.55);
    }

    footer, #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    with open("personality_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact


@st.cache_data
def load_dataset():
    df = pd.read_csv("personality_dataset.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


try:
    artifact = load_model()
    model = artifact["model"]
    encoders = artifact["encoders"]
    feature_order = artifact["feature_order"]
    model_accuracy = artifact["accuracy"]
except FileNotFoundError:
    st.error("⚠️ File 'personality_model.pkl' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
    st.stop()

target_encoder = encoders["Personality"]

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Personality Predictor</div>
    <div class="hero-subtitle">Prediksi tipe kepribadian Introvert atau Extrovert menggunakan Machine Learning (Random Forest)</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini menggunakan model **Random Forest Classifier** "
        "yang dilatih pada dataset kepribadian untuk memprediksi apakah "
        "seseorang cenderung **Introvert** atau **Extrovert** berdasarkan "
        "kebiasaan sosial sehari-hari."
    )
    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    st.metric("Akurasi pada Data Uji", f"{model_accuracy*100:.2f}%")
    st.markdown("---")
    st.markdown("### 🗂️ Menu")
    page = st.radio("Pilih halaman:", ["🔮 Prediksi", "📈 Eksplorasi Data", "ℹ️ Tentang Model"])
    st.markdown("---")
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit & Scikit-learn")

# =========================================================
# HALAMAN: PREDIKSI
# =========================================================
if page == "🔮 Prediksi":
    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Masukkan Data Kebiasaan Sosial")

        c1, c2 = st.columns(2)
        with c1:
            time_alone = st.slider(
                "⏰ Waktu Dihabiskan Sendirian (jam/hari)",
                min_value=0.0, max_value=11.0, value=4.0, step=1.0,
                help="Rata-rata jam per hari yang dihabiskan sendirian"
            )
            social_event = st.slider(
                "🎉 Frekuensi Hadir di Acara Sosial",
                min_value=0.0, max_value=10.0, value=4.0, step=1.0,
                help="Skala frekuensi kehadiran pada acara sosial"
            )
            going_outside = st.slider(
                "🚶 Frekuensi Pergi Keluar Rumah",
                min_value=0.0, max_value=7.0, value=3.0, step=1.0,
                help="Skala frekuensi pergi ke luar rumah"
            )
            friends_circle = st.slider(
                "👥 Jumlah Lingkar Pertemanan",
                min_value=0.0, max_value=15.0, value=6.0, step=1.0,
                help="Jumlah teman dekat dalam lingkar sosial"
            )

        with c2:
            post_freq = st.slider(
                "📱 Frekuensi Posting di Media Sosial",
                min_value=0.0, max_value=10.0, value=4.0, step=1.0,
                help="Skala frekuensi posting di media sosial"
            )
            stage_fear = st.radio(
                "😰 Apakah Anda Memiliki Rasa Takut Tampil di Depan Umum?",
                ["No", "Yes"],
                horizontal=True,
            )
            drained = st.radio(
                "🔋 Apakah Anda Merasa Lelah Setelah Bersosialisasi?",
                ["No", "Yes"],
                horizontal=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🔮 Prediksi Kepribadian Saya")

    with col_result:
        if predict_btn:
            stage_fear_enc = encoders["Stage_fear"].transform([stage_fear])[0]
            drained_enc = encoders["Drained_after_socializing"].transform([drained])[0]

            input_dict = {
                "Time_spent_Alone": time_alone,
                "Stage_fear": stage_fear_enc,
                "Social_event_attendance": social_event,
                "Going_outside": going_outside,
                "Drained_after_socializing": drained_enc,
                "Friends_circle_size": friends_circle,
                "Post_frequency": post_freq,
            }
            input_df = pd.DataFrame([input_dict])[feature_order]

            pred_encoded = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            pred_label = target_encoder.inverse_transform([pred_encoded])[0]

            classes = target_encoder.classes_
            proba_dict = dict(zip(classes, pred_proba))
            confidence = proba_dict[pred_label] * 100

            if pred_label == "Extrovert":
                st.markdown(f"""
                <div class="result-extrovert">
                    <div class="result-emoji">🤟😝🤟</div>
                    <div class="result-label">EXTROVERT</div>
                    <div class="result-desc">Anda cenderung energik, suka bersosialisasi, dan nyaman menjadi pusat perhatian!</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-introvert">
                    <div class="result-emoji">😨</div>
                    <div class="result-emoji">👉     👈</div>
                    <div class="result-label">INTROVERT</div>
                    <div class="result-desc">Anda cenderung tenang, reflektif, dan mengisi ulang energi lewat waktu sendiri!</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge confidence
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'suffix': "%", 'font': {'color': 'white', 'size': 36}},
                title={'text': "Tingkat Keyakinan Model", 'font': {'color': 'white', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                    'bar': {'color': "#E100FF"},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255,255,255,0.06)'},
                        {'range': [50, 100], 'color': 'rgba(255,255,255,0.12)'},
                    ],
                }
            ))
            fig.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Probability breakdown
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 📊 Detail Probabilitas")
            prob_df = pd.DataFrame({
                "Kepribadian": classes,
                "Probabilitas (%)": pred_proba * 100
            }).sort_values("Probabilitas (%)", ascending=False)

            fig_bar = px.bar(
                prob_df, x="Probabilitas (%)", y="Kepribadian", orientation="h",
                color="Kepribadian",
                color_discrete_map={"Extrovert": "#FFC837", "Introvert": "#6dd5ed"},
                text="Probabilitas (%)",
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_bar.update_layout(
                showlegend=False,
                height=200,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                xaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.1)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 3rem 1.5rem;">
                <div style="font-size: 3.5rem;">👈</div>
                <h3>Isi data di sebelah kiri</h3>
                <p style="color:#cfc7e8;">Kemudian klik tombol <b>Prediksi Kepribadian Saya</b> untuk melihat hasilnya di sini.</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# HALAMAN: EKSPLORASI DATA
# =========================================================
elif page == "📈 Eksplorasi Data":
    df = load_dataset()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Eksplorasi Dataset Personality")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-pill"><div class="value">{len(df)}</div><div class="label">Total Data</div></div>""", unsafe_allow_html=True)
    with m2:
        n_extro = (df["Personality"] == "Extrovert").sum()
        st.markdown(f"""<div class="metric-pill"><div class="value">{n_extro}</div><div class="label">Extrovert</div></div>""", unsafe_allow_html=True)
    with m3:
        n_intro = (df["Personality"] == "Introvert").sum()
        st.markdown(f"""<div class="metric-pill"><div class="value">{n_intro}</div><div class="label">Introvert</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-pill"><div class="value">{df.shape[1]-1}</div><div class="label">Jumlah Fitur</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🎯 Distribusi Target")
        fig_pie = px.pie(
            df, names="Personality", hole=0.45,
            color="Personality",
            color_discrete_map={"Extrovert": "#FFC837", "Introvert": "#6dd5ed"},
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
            height=320, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🔥 Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Fitur": feature_order, "Importance": importances
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Fitur", orientation="h",
            color="Importance", color_continuous_scale="Plasma",
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}, height=320, showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 📊 Distribusi Fitur Numerik berdasarkan Kepribadian")
    num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
                "Friends_circle_size", "Post_frequency"]
    selected_feature = st.selectbox("Pilih fitur untuk dilihat distribusinya:", num_cols)
    fig_hist = px.histogram(
        df, x=selected_feature, color="Personality", barmode="overlay",
        color_discrete_map={"Extrovert": "#FFC837", "Introvert": "#6dd5ed"},
        opacity=0.75, nbins=20,
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}, height=350,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 🔍 Lihat Data Mentah")
    st.dataframe(df, use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HALAMAN: TENTANG MODEL
# =========================================================
elif page == "ℹ️ Tentang Model":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Tentang Model & Cara Kerja")
    st.markdown("""
**Algoritma:** Random Forest Classifier

**Fitur yang digunakan:**
1. ⏰ **Time_spent_Alone** — Waktu yang dihabiskan sendirian (jam/hari)
2. 😰 **Stage_fear** — Memiliki rasa takut tampil di depan umum (Yes/No)
3. 🎉 **Social_event_attendance** — Frekuensi menghadiri acara sosial
4. 🚶 **Going_outside** — Frekuensi pergi keluar rumah
5. 🔋 **Drained_after_socializing** — Merasa lelah setelah bersosialisasi (Yes/No)
6. 👥 **Friends_circle_size** — Jumlah lingkar pertemanan
7. 📱 **Post_frequency** — Frekuensi posting di media sosial

**Target:** Personality (Extrovert / Introvert)
    """)
    st.markdown(f"**Akurasi pada data uji:** `{model_accuracy*100:.2f}%`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 🌳 Parameter Model")
    params = model.get_params()
    show_params = {k: params[k] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"]}
    st.json(show_params)
    st.markdown("</div>", unsafe_allow_html=True)

    
