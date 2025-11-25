import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from collections import Counter
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Transportasi Jakarta",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model HuggingFace
@st.cache_resource
def load_sentiment_model():
    model_name = "w11wo/indobert-large-p1-twitter-indonesia-sarcastic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Fungsi untuk analisis sentimen
def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Mapping kelas: 0=negative, 1=neutral, 2=positive (sesuaikan dengan model Anda)
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    confidence = predictions[0][predicted_class].item()
    
    return sentiment_labels[predicted_class], confidence

# Fungsi untuk deteksi problem
def detect_problems(text):
    problems_keywords = {
        'Keterlambatan': ['lama', 'tunggu', 'telat', 'ngaret', 'delay', 'nantri', 'antri', 'jam'],
        'Kondisi': ['panas', 'bau', 'kotor', 'penuh', 'sesak', 'padat', 'rusak', 'sobek', 'kumuh'],
        'Pelayanan': ['sopir', 'kondektur', 'staff', 'karyawan', 'pelayan', 'ramah', 'galak', 'marah'],
        'Navigasi': ['salah', 'nyasar', 'turun', 'henti', 'rute', 'jalur', 'arah', 'peta'],
        'Kemacetan': ['macet', 'padat', 'lalu lintas', 'jalanan', 'tertahan'],
        'Masalah Pembayaran': ['bayar', 'tarif', 'harga', 'uang', 'gratis', 'mahal', 'murah', 'saldo'],
        'Akses/Rute': ['akses', 'rute', 'jalur', 'trayek', 'lewati', 'turun', 'naik'],
        'Emosi/Frustrasi': ['kesal', 'marah', 'frustrasi', 'jengkel', 'sebel', 'gemes', 'stress'],
        'Infrastruktur': ['halte', 'stasiun', 'bangunan', 'fasilitas', 'toilet', 'kursi'],
        'Keamanan': ['aman', 'selamat', 'jahat', 'copet', 'pencuri', 'keamanan'],
        'Kenyamanan': ['nyaman', 'enak', 'legit', 'adem', 'sejuk', 'bersih'],
        'Harga': ['murah', 'mahal', 'harga', 'tarif', 'biaya', 'ongkos']
    }
    
    detected_problems = []
    text_lower = text.lower()
    
    for problem, keywords in problems_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_problems.append(problem)
                break
    
    return list(set(detected_problems))

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .tweet-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .problem-tag {
        background-color: #e9ecef;
        padding: 2px 8px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<div class="main-header">🚍 Dashboard Analisis Sentimen Transportasi Jakarta</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Ganti dengan path file CSV Anda
    try:
        df = pd.read_csv("final_df.csv")
    except:
        # Fallback data sample jika file tidak ditemukan
        data = {
            'Kategori': ['jak', 'jak', 'tj', 'tj', 'krl', 'krl'],
            'Tweet': [
                'Jaklingko sangat nyaman dan tepat waktu hari ini',
                'Saya menunggu Jaklingko terlalu lama, keterlambatan yang menyebalkan',
                'Transjakarta AC-nya dingin dan sopirnya ramah',
                'Bus Transjakarta penuh sesak dan tidak nyaman',
                'KRL hari ini berjalan dengan lancar dan nyaman',
                'KRL sangat padat dan berisik, tidak nyaman'
            ],
            'Sentiment': ['Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif'],
            'problem': [
                "['Kenyamanan']",
                "['Keterlambatan', 'Emosi/Frustrasi']",
                "['Kenyamanan', 'Pelayanan']",
                "['Kondisi', 'Kenyamanan']",
                "['Kenyamanan']",
                "['Kondisi', 'Kenyamanan']"
            ]
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# Preprocess problem data
def preprocess_problems(problem_str):
    if pd.isna(problem_str) or problem_str == '[]':
        return []
    try:
        # Clean and parse the problem string
        problems = eval(problem_str) if isinstance(problem_str, str) else problem_str
        return [p.strip() for p in problems] if isinstance(problems, list) else []
    except:
        return []

df['problems_clean'] = df['problem'].apply(preprocess_problems)

# Load model
tokenizer, model = load_sentiment_model()

# Sidebar untuk input analisis
with st.sidebar:
    st.markdown("### 🔍 Analisis Komentar Baru")
    
    new_comment = st.text_area(
        "Masukkan komentar tentang transportasi:",
        placeholder="Contoh: 'Jaklingko hari ini sangat nyaman dan tepat waktu'",
        height=100
    )
    
    transport_category = st.selectbox(
        "Pilih kategori transportasi:",
        ["jak", "tj", "krl"],
        format_func=lambda x: {"jak": "JakLingko", "tj": "TransJakarta", "krl": "KRL"}[x]
    )
    
    analyze_btn = st.button("Analisis Komentar", type="primary")

# Proses analisis komentar baru
if analyze_btn and new_comment:
    with st.spinner("Menganalisis sentimen dan masalah..."):
        # Analisis sentimen
        sentiment, confidence = analyze_sentiment(new_comment, tokenizer, model)
        
        # Deteksi masalah
        problems = detect_problems(new_comment)
        
        # Tampilkan hasil
        st.success("Analisis selesai!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_color = {
                "Positif": "sentiment-positive",
                "Negatif": "sentiment-negative", 
                "Netral": "sentiment-neutral"
            }
            st.markdown(f'<div class="metric-card"><h3>Sentimen</h3><p class="{sentiment_color[sentiment]}">{sentiment}</p><p>Confidence: {confidence:.2f}</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Kategori</h3><p>{"JakLingko" if transport_category == "jak" else "TransJakarta" if transport_category == "tj" else "KRL"}</p></div>', unsafe_allow_html=True)
        
        with col3:
            problems_text = ", ".join(problems) if problems else "Tidak terdeteksi"
            st.markdown(f'<div class="metric-card"><h3>Masalah Terdeteksi</h3><p>{problems_text}</p></div>', unsafe_allow_html=True)
        
        # Tambahkan ke dataset sementara
        new_data = {
            'Kategori': transport_category,
            'Tweet': new_comment,
            'Sentiment': sentiment,
            'problem': str(problems),
            'problems_clean': problems
        }
        
        # Update dataframe
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        
        st.info("Data telah ditambahkan ke dataset sementara. Visualisasi akan diperbarui.")

# Tabs untuk masing-masing transportasi
tab1, tab2, tab3 = st.tabs(["🚐 JakLingko", "🚍 TransJakarta", "🚆 KRL"])

def create_transport_tab(category, category_name):
    # Filter data berdasarkan kategori
    category_data = df[df['Kategori'] == category].copy()
    
    if category_data.empty:
        st.warning(f"Tidak ada data untuk {category_name}")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tweets = len(category_data)
    positive_tweets = len(category_data[category_data['Sentiment'] == 'Positif'])
    negative_tweets = len(category_data[category_data['Sentiment'] == 'Negatif'])
    neutral_tweets = len(category_data[category_data['Sentiment'] == 'Netral'])
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Tweet</h3><h2>{total_tweets}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Positif</h3><h2 style="color: #28a745;">{positive_tweets}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Negatif</h3><h2 style="color: #dc3545;">{negative_tweets}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Netral</h3><h2 style="color: #ffc107;">{neutral_tweets}</h2></div>', unsafe_allow_html=True)
    
    # Visualisasi 1: Top Problems
    st.markdown(f'<div class="sub-header">📊 Top 5 Masalah pada {category_name}</div>', unsafe_allow_html=True)
    
    # Extract all problems
    all_problems = []
    for problems in category_data['problems_clean']:
        all_problems.extend(problems)
    
    if all_problems:
        problem_counts = Counter(all_problems)
        top_problems = problem_counts.most_common(5)
        
        problems_df = pd.DataFrame(top_problems, columns=['Problem', 'Count'])
        
        fig = px.bar(
            problems_df, 
            x='Count', 
            y='Problem',
            orientation='h',
            title=f'Top 5 Masalah {category_name}',
            color='Count',
            color_continuous_scale='blues'
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada data masalah yang terdeteksi")
    
    # Visualisasi 2: Sentiment Distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<div class="sub-header">📈 Distribusi Sentimen</div>', unsafe_allow_html=True)
        
        sentiment_counts = category_data['Sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title=f'Distribusi Sentimen {category_name}',
            color=sentiment_counts.index,
            color_discrete_map={
                'Positif': '#28a745',
                'Negatif': '#dc3545', 
                'Netral': '#ffc107'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="sub-header">🔍 Tren Masalah</div>', unsafe_allow_html=True)
        
        if all_problems:
            # Problem frequency table
            problem_freq = pd.DataFrame(problem_counts.most_common(10), columns=['Problem', 'Frekuensi'])
            st.dataframe(problem_freq, use_container_width=True, height=300)
        else:
            st.info("Belum ada data masalah")
    
    # Tweet terbaru
    st.markdown(f'<div class="sub-header">💬 Tweet Terbaru tentang {category_name}</div>', unsafe_allow_html=True)
    
    recent_tweets = category_data.tail(10)
    
    for _, tweet in recent_tweets.iterrows():
        sentiment_class = f"sentiment-{tweet['Sentiment'].lower()}"
        
        problems_html = ""
        if tweet['problems_clean']:
            problems_html = "<div style='margin-top: 10px;'>"
            for problem in tweet['problems_clean']:
                problems_html += f'<span class="problem-tag">{problem}</span>'
            problems_html += "</div>"
        
        tweet_html = f"""
        <div class="tweet-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <p style="margin: 0; font-size: 0.9rem;">{tweet['Tweet']}</p>
                    {problems_html}
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <p class="{sentiment_class}" style="margin: 0; font-size: 0.8rem;">{tweet['Sentiment']}</p>
                </div>
            </div>
        </div>
        """
        
        st.markdown(tweet_html, unsafe_allow_html=True)

# Isi masing-masing tab
with tab1:
    create_transport_tab('jak', 'JakLingko')

with tab2:
    create_transport_tab('tj', 'TransJakarta')

with tab3:
    create_transport_tab('krl', 'KRL')

# Footer
st.markdown("---")
st.markdown(
    "Dashboard Analisis Sentimen Transportasi Jakarta | "
    "Data diperbarui secara real-time dengan analisis AI"
)