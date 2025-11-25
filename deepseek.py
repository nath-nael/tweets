import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from collections import Counter
import numpy as np
import ast

# Fix for numpy compatibility
try:
    np.bool8 = np.bool_  # For compatibility with older code
except:
    pass

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
    try:
        model_name = "w11wo/indobert-large-p1-twitter-indonesia-sarcastic"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Fungsi untuk analisis sentimen (fallback jika model tidak bisa load)
def analyze_sentiment(text, tokenizer, model):
    if tokenizer is None or model is None:
        # Fallback simple sentiment analysis
        negative_words = ['lama', 'tunggu', 'telat', 'macet', 'penuh', 'rusak', 'jelek', 'buruk', 'sebel', 'kesal', 'marah', 'frustrasi']
        positive_words = ['bagus', 'baik', 'nyaman', 'cepat', 'murah', 'puas', 'senang', 'recommend', 'enak', 'mantap']
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if positive_count > negative_count:
            return "Positif", 0.7
        elif negative_count > positive_count:
            return "Negatif", 0.7
        else:
            return "Netral", 0.5
    
    # Original model-based analysis
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Mapping kelas: 0=negative, 1=neutral, 2=positive
    sentiment_labels = ["Negatif", "Netral", "Positif"]
    confidence = predictions[0][predicted_class].item()
    
    return sentiment_labels[predicted_class], confidence

# Fungsi untuk deteksi problem
def detect_problems(text):
    problem_keywords= {
    'Keterlambatan': [
        'telat', 'terlambat', 'lambat', 'delay', 'nunggu', 'nungguin', 'ga dateng2', 'gk dtg dtg', 'gak nyampe',
        'gk dateng', 'ngetem', 'molor', 'lama banget', '1 jam lebih', 'kena delay', 'datengnya lama',
        'nunggu sejam', 'kelamaan', 'ga jalan-jalan', 'jadwal ngaco', 'jadwal gak jelas',
        'jamnya gak sesuai', 'terhambat', 'lelet', 'ngaret', 'lama bgt', 'nunggu lama',
        'telat parah', 'datangnya lama', 'schedule ngaco', 'kepagian', 'kemaleman',
        'lebih dari 1 jam', 'berjam-jam', 'ngaret parah', 'nunggu lama sekali',
        'selesai lama', 'kosong lama', 'datang lama', 'jaklingko nya lama',
        'duluan siapa yak', 'nunggu jaklingko 07 adl hal ter sial lamaa bgt','ga dateng', 'belum dateng',
        'nunggu bus', 'nungguin lama', 'nunggu di halte', 'gak nyampe-nyampe',
        'ga jalan', 'berhenti lama', 'stuck di jalan', 'rutenya lama banget', 'nunggu kereta lewat',
        'nunggu busnya dateng', 'nunggu giliran', 'nunggu lama banget', 'nungguin transjakarta',
        'nunggu kereta', 'nunggu lama parah', 'ngetem mulu', 'ga gerak-gerak', 'nunggu bus doang', # BARU
        'nunggu busnya lama', 'nungguin jaklingko', 'nunggu giliran naik',
        'nunggu kebanyakan', 'nunggu di bawah hujan', 'nunggu dari tadi',
        'nungguin ga dateng-dateng', 'nunggu 1 jam',
        'busnya lama banget', 'nunggu antrian panjang', 'nunggu di pinggir jalan',
        'menunggu lama', 'tunggu lama banget', '1 jam 30 menit', 'busnya lama dateng', 'ngetem mulu',
        'nunggu lama', 'nungguin lama banget', 'nunggu sejam', 'kena delay', 'dateng lama banget',
        'jadwal ngaco', 'datangnya lama', 'ngetem lama', 'nunggu giliran', 'nunggu kejadian lama', 'nunggu lebih dari 1 jam',
        'nunggu lama', 'telat banget', 'ngetem mulu', 'ngaret', 'kena delay', 'lambat banget',
        '1 jam lebih', 'datangnya lama', 'ngaret parah', 'telat parah', 'nunggu giliran lama',
        'datangnya telat', 'molor parah', 'nunggu bus lama', 'susah nyampe', "terlambat banget",
        'lama nunggu', 'nunggunya lama', 'nunggu doang', 'nunggu dari tadi banget',
        'nunggu dari jam', 'nunggu udah lama', 'nunggu terus', 'nunggunya lama banget',
        'udah nunggu', 'nunggu dari pagi', 'nungguin doang', 'nunggu setengah jam',
        'ga nyampe-nyampe', 'nunggu kaga dateng', 'nunggu ga muncul', 'nunggu kaga nyampe',
        'nunggu kaga dateng-dateng', 'nunggu kaga jalan', 'nunggu kaga berangkat','mana jaklingko',
        "nunggu", "lama tj", "perlu waktu", "terlambat", "tunggu lama", 'buruan dateng jaklingko'
        ,'nunggu jaklingko', '1 JAM 30 MENIT','jalan nya lama','lama begini',
        'ga lewat2','siklus kedatangan bus gak sehat','lumayan lama','4 kereta jurusan bogor udah lewat tapi tujuan jakartakota belum 1 pun',
        'molor', 'telat banget', 'terlambat parah', 'jadwal ngaco', 'datang terlambat', 'nunggu kelamaan',
        'sempat delay', 'sempat terlambat', 'nunggu terlalu lama', 'sudah datang terlambat', 'kereta telat',
        'terhambat waktu', 'schedule kacau', 'terlambat parah', 'waktu yang tidak tepat', 'keterlambatan panjang',
        'keterlambatan luar biasa', 'lebih lama dari estimasi','jaklingko lama', 'nunggu jaklingko lama', 'nunggu bus jaklingko',
        'delay banget', 'gagal naik jaklingko','nungguin jaklingko terus', 'nunggu lama banget', 'jaklingko 1 jam', 'nunggu bus lewat', 'nunggu terlalu lama',
        'nunggu 2 jam lebih', 'nunggu 3 jam', 'nunggu terlalu lama giliran jaklingko'


    ],

    'Pelayanan': [
        'driver', 'sopir', 'supir', 'kasar', 'jutek', 'marah-marah', 'teriak', 'tidak sopan', 'ngomel',
        'ngamuk', 'pelayanan jelek', 'crew nya galak', 'ngebentak', 'pelayanan buruk',
        'sopir nyebelin', 'ga ramah', 'cuek banget', 'ogah ngeladenin', 'dikasarin', 'pelayanan parah',
        'petugas', 'kondektur', 'judes', 'sok iye', 'nyolot', 'ngeselin', 'ga ada akhlak',
        'rese', 'pegawai', 'staff', 'gaje', 'gak profesional', 'malas', 'minta tips',
        'minta uang', 'main hp', 'ngobrol', 'tidak membantu', 'kurang ajar', 'sensian',
        'bapaknya', 'galak', 'perilaku pengemudi', 'sinis', 'ga ditengok samsek', # BARU: Gak digubris/diperhatikan
        'ga dibolehin', 'ditengok', 'kurang ajar','ga direspon', 'gak digubris', 'gak ditolong', 'gak diladenin',
        'dicuekin', 'sopir ugal','petugas tidur', 'ga tanggap', 'ga bantu', 'sopir ngebut', 'sopir ngawur', 'ga peduli',
        'ga nunggu penumpang', 'nyolot banget', 'petugas malas', 'ga profesional banget', 'gak care',
        'petugasnya diem aja', 'ga sabar', 'sopir marah', 'driver cuek',# BARU
        'supir diem aja', 'sopir ga nyapa', 'sopir diem doang',
        'sopir jutek', 'supir galak', 'sopir ga bantu', 'ga dikasih info',
        'sopir ga peduli', 'petugas ga bantu', 'supir nyolot', 'petugas diem aja',
        'sopir cuek', 'ngamuk', 'berperilaku buruk', 'galak', 'sopir ngebut',
        'tidak sopan', 'ngomel', 'bapaknya galak', 'petugas malas', 'sopir malas', 'petugas tidak ramah',
        'ga guna', 'gak guna', 'ga becus', 'gak becus', 'pelayanan zonk', 'ga ada yang bantu',
        'petugasnya kemana', 'sopirnya kemana', 'ga dijelasin', 'ga dikasih tau',
        'sopirnya ilang', 'sopir ga muncul', 'petugasnya ga ada', 'staff ga standby', 'sensian','ngebut-ngebutan','"sopir curiga',
        'kalo aja jaklingko','asli enek bgt','bikin kesel','jalan belok','ugal ugalan','grasak grusuk','langsung habis tiketnya',
        'volume speaker yang dirasa terlalu kencang',
        'berdesakan', 'overload', 'sumpek', 'kurang nyaman', 'ac lemah',
        'tidak profesional', 'pelayanan buruk', 'pelayanan sangat mengecewakan', 'sopir nyebelin',
        'crew galak', 'petugas tidak ramah', 'tidak ada bantuan', 'petugas cuek', 'sopir tidak sabar',
        'tidak ada yang melayani', 'ga ada perhatian', 'petugas sibuk sendiri', 'pelayanan mengecewakan',
        'staff cuek', 'pengemudi tidak ramah', 'pelayanan tak memuaskan', 'kurang tanggap','penjaga tj mya sama sekali gaa sollutif',
        'staff jaklingko ga ramah', 'sopir jaklingko ga peduli', 'pelayanan jaklingko buruk', 'crew jaklingko malas',
        'petugas jaklingko ga tanggap', 'sopir jaklingko nyolot', 'petugas jaklingko diem aja', 'petugas jaklingko kurang sopan',
        'sopir jaklingko ngebut', 'petugas jaklingko tidur', 'pelayanan sopir jaklingko jelek', 'staff jaklingko galak'

    ],

    'Kondisi': [
        'rusak', 'ac mati', 'ac rusak', 'panas', 'gerah', 'bau', 'bau apek', 'kotor', 'jorok', 'becek',
        'karatan', 'kursinya copot', 'pecah', 'busnya jelek', 'udah tua', 'nggak bersih',
        'sempit', 'penuh', 'padat', 'berdesakan', 'ga muat', 'overload', 'sesak', 'ngap-ngapan',
        'rembes', 'basah', 'nggak layak', 'ga nyaman', 'tidak layak', 'sumpek', 'pengap',
        'kain kotor', 'jendela pecah', 'bus reyot', 'bus rombeng', 'bus rongsok', 'berkarat',
        'jijik', 'bau pesing', 'parah kondisinya', 'kaca pecah', 'pintu macet', 'kursi sobek',
        'jendela kotor', 'sarang laba-laba', 'sampah', 'kumuh', 'bising', 'asap', 'pengap',
        'jendela susah', 'jendela jaklingko tuh susah banget', 'pintunya nutup', 'dingin bgt',
        'spek beton k500', 'kejepit','penuh banget', 'desek-desekan', 'ga kebagian duduk', 'berdiri terus', 'bau badan',
        'bau asap', 'bau bensin', 'pengap banget', 'panas banget', 'kursi keras', 'pintu susah dibuka',
        'pintu ga kebuka', 'pintu error', 'ac ga nyala', 'ac lemah', 'ruang sempit', 'semrawut',
        'rem mendadak', 'ngebut banget', 'guncangannya parah', 'ga nyaman banget', 'kotor banget',# BARU
        'bus penuh terus', 'padet banget', 'semrawut banget', 'bus kecil banget',
        'nabrak-nabrak', 'rem mendadak banget', 'bau bensin banget', 'ga muat berdiri',
        'kursinya keras banget', 'bus sempit', 'penumpang berdiri semua', 'bau solar',
        'bus tua', 'berisik banget', 'guncangan parah', 'jalan ga mulus', 'bus ringkih',
        'ac mati', 'bus rusak', 'kotor banget', 'kursi rusak', 'bus reyot', 'jalur sempit', 'nggak nyaman',
        'kursi keras', 'rem blong', 'jalan berlobang', 'kursinya sobek', 'berdesakan', 'padat', 'nggak layak',
        'kayak sauna', 'kayak neraka', 'ampas banget', 'bus ampas', 'bau banget',
        'kursi udah hancur', 'kursinya keras banget', 'busnya panas', 'kayak oven',
        'kayak kandang', 'gerah banget', 'busnya parah', 'kondisi parah','tj dingin bgt','ga enak rasanya',
        'mual di tj','kecill bangett','ac krl jakarta kota mati','acnya ga nyala',
        'susah dpt jaklingko','sering kejedot pintu jaklingko','tempatnya beneran kecil',
        'nih kursi kek lagi naik kereta ekonomi','tj brutal bgt','ga dapat kursi di tj','naik tj lebih susah haltenya',
        'weekend gini penumpang banyak','kursi nggak ada','kalo lowdeck sering gini terutama yg listrik',
        '8 spot healing dekat jakarta','kalo buru buru pake gojek kalo nyantai banget ikut jaklingko',
        'sepadet krl jakarta kotabogor di jam rush hour','aku pernah naik kereta sendiri 7 jam jogjajakarta',
        'kereta reyot', 'bus kotor', 'kondisi bus jelek', 'kereta penuh', 'kondisi fasilitas buruk',
        'fasilitas rusak', 'kereta berdebu', 'bus sempit', 'kondisi kendaraan parah', 'bus tidak layak',
        'kursi tidak nyaman', 'kursi rusak', 'ruang sempit', 'kereta ringkih', 'terlalu penuh', 'mobil reyot',
        'jalanan rusak', 'kereta tidak layak', 'pintu susah dibuka', 'kursi keras', 'sarana tidak nyaman',
        '8 jam kalo naik kereta jarak jauh','kenapaa schedule semua ini transjakartanyaa',
        'mengeluh bila tj error','emang langka aje busnya drtd juga yg lewat', 'penuh', 'semrawut',
        'naik jaklingko seangkot ama pengamen''penuh sesak', 'berdesakan', 'terhalang','kurang terorganisir', 'bising', 'terganggu', 'sesak', 'keramaian',
        'kesulitan bergerak','bus jaklingko sempit', 'mobil jaklingko rusak', 'jaklingko sesak', 'bus jaklingko penuh', 'kerusakan jaklingko',
        'bus jaklingko bau', 'kursi jaklingko rusak', 'ac jaklingko mati', 'kereta jaklingko panas', 'jaklingko kotor',
        'bus jaklingko penuh sesak', 'kursi bus jaklingko keras', 'jaklingko sempit banget','karena ngga ada jaklingko',
        'ga kuat itu pasti nanti antrenya','gabisa ditutup pintunya gegara pada maksa masuk',
        'kemana saja jaklingko30 arah balik grogol tobar dll antri panjang untung',


    ],

    'Masalah Pembayaran': [
        'mahal', 'kemahalan', 'tap error', 'saldo ilang', 'saldo kesedot', 'saldo kepotong',
        'tap gagal', 'emoney', 'uang', 'duit', 'bayar', 'beli kartu', 'topup', 'top up',
        'e-money', 'gagal tap', 'error tap', 'kesusahan bayar', 'harga naik', 'ga bisa tap',
        'susah bayar', 'sistem pembayaran aneh', 'gagal transaksi', 'gagal bayar',
        'taping', 'kartu error', 'tap in', 'tap out', 'tarif', 'uang elektronik', 'naik harga',
        'kurang saldo', 'tidak ada kembalian', 'kembalian', 'uang pas', 'gopay', 'ovo', 'dana',
        'qr code', 'aplikasi error', 'aplikasi', 'salah potong', 'motong saldo 2x',
        'kartu jaklingko gw kecuci anyink jd pengok', 'kartu lain yaa', 'kartunyaa', # BARU: Isu fisik/jenis kartu
        'ngetap pertama 3500 ngetap kedua 1500 ngetap ketiga 0', '5000 pake kartu jaklingko',# BARU: Isu tarif/skema bayar
        'tap gagal', 'saldo kepotong dua kali', 'tap error lagi', 'kartu rusak', 'kartu ga kebaca',
        'gabisa tap', 'saldo ilang', 'saldo error', 'harga beda', 'bayarnya double',
        'tarif ga sesuai', 'salah potong saldo', 'topup error', 'qr error', 'tap in gagal', 'tap out gagal',
        'kartu jaklingko error', 'ngetap tiga kali', 'gabisa scan', 'scanner error',
        'kartu ilang', 'kartu ketinggalan', 'kartu rusak total', 'kartu jaklingko ilang',
        'kartu gak kebaca', 'saldo minus', 'harga aneh', 'tap error lagi', 'tapnya lama banget',
        'kartu jaklingko abis masa aktif', 'scanner rusak', 'kartu double tap',
        'tarif KRL bayar dua kali', 'saldo minus', 'kartu tidak ter‑scan', 'top‑up gagal', 'integrasi tarif salah',
        'error saldo', 'bayar double', 'tap error lagi', 'saldo kesedot', 'salah potong saldo',
        'kurang saldo', 'saldo dipotong 2x', 'kartu jaklingko error', 'sistem pembayaran gangguan', 'aplikasi error bayar',
        'tap tapan', 'scanner ngaco', 'kartu gabisa dibaca', 'kartu ga ke-scan',
        'mesin error', 'tap lama', 'kartu ga respon', 'mesinnya rusak', 'alat tap error',
        'saldo ga balik', 'saldo ga masuk', 'saldo kepotong tapi gagal', 'scanner mati',
        "saldo", "motong saldo", "bayar", "emoney", "potong", "tarif", "harga",'cash out jaklingko','gabisa kebaca',
        'harga mahal', 'bayar lebih', 'saldo tidak cukup', 'harga tidak sesuai', 'bayar dua kali',
        'gagal bayar', 'sistem pembayaran error', 'harga naik mendadak', 'kartu pembayaran rusak',
        'gagal tap', 'harga gak jelas', 'saldo hilang', 'minta biaya tambahan', 'pembayaran double',
        'harga bertambah', 'membayar lebih', 'kartu tidak terdeteksi', 'harga tidak transparan',
        'error di sistem pembayaran', 'saldo tiba-tiba hilang', 'harga yang aneh',
        'kartu jaklingko saya dari bank dki kok mendadak ga bisa digunakan 3 harian ini ya',
        'gagal bayar jaklingko', 'saldo jaklingko hilang', 'topup jaklingko gagal', 'saldo jaklingko kesedot',
        'kartu jaklingko rusak', 'gagal tap jaklingko', 'saldo jaklingko hilang', 'aplikasi jaklingko error',
        'tap jaklingko gagal', 'saldo jaklingko berkurang', 'topup jaklingko tidak masuk','di bus manapun tapnya gagal',


    ],

    'Navigasi': [
        'bingung', 'ga jelas', 'rute aneh', 'rute ribet', 'rutenya ngaco', 'ga ngerti', 'susah naik',
        'turun di mana', 'naik di mana', 'berhenti tiba-tiba', 'salah turun', 'salah naik',
        'rute muter-muter', 'salah jalur', 'nyasar', 'ketinggalan halte', 'bingung rutenya',
        'petunjuk ga jelas', 'nggak tau harus kemana', 'arah gak jelas', 'bikin bingung',
        'papan petunjuk', 'halte ga ada', 'berhenti mendadak', 'jalur salah', 'muter-muter',
        'peta', 'signage', 'informasi salah', 'tidak ada pengumuman', 'lewati halte',
        'lewatin', 'putus di tengah jalan', 'jalur transit', 'missed my stop', 'bablasan halte',
        'lupa bilang stop depan pak', 'belom kelewat', # BARU: Masalah pemberhentian (stop request)
        'lewat halte', 'bablas halte', 'turun salah', 'turun jauh', 'berhenti di seberang',
        'salah jalan', 'muter terus', 'ga berhenti depan', 'berhenti jauh dari halte',
        'stop kelewatan', 'turun sebelum halte', 'salah arah', 'nyasar', 'ga jelas rutenya',
        'muter2', 'belok salah', 'salah stop', 'lewat terus',
        'salah halte', 'salah arah bus', 'ga ngerti rutenya', 'ga tau harus turun di mana',
        'halte kelewatan', 'turun di tempat salah', 'ga berhenti depan halte', 'muter-muter terus',
        'kelewat halte', 'nyasar ke arah lain', 'turun di tempat jauh',
        'bingung banget', 'rutenya aneh', 'bingung mau naik apa', 'bingung mau turun di mana',
        'rute ga nyambung', 'jalur ga jelas banget', 'bingung arah', 'ga paham jalurnya','bingung lah',
        'nyasar','gak ada aba-aba','susah banget','nyetopnya gimana','muternya jauh',
        'ribet abis','kamu gratis minusnya tiap naik gw harus slalu on gmaps supaya gak kelewat krn gak ada aba2',
        'infokan jaklingko arah lenteng agung ke ajinomoto dong','krl tujuan jakarta kota dari arah cikarangbekasi','naik tj mutermuter',
        'sebagai penumpang gmn cara make jaklingko nih meng', 'rute muter', 'akses yang sulit', 'rute tidak jelas', 'turun di tempat salah', 'bingung mencari jalur',
        'rute ga nyambung', 'petunjuk yang tidak jelas', 'bikin salah turun', 'turun salah halte', 'jalur tidak jelas',
        'rute yang bingung', 'halternya gak ada', 'belok salah', 'jalan yang tidak terhubung', 'tidak ada informasi yang jelas',
        'rute penuh', 'nyasar karena tidak ada rambu', 'informasi halte ga jelas', 'muter-muter ga jelas',
        'rute jaklingko ngaco', 'jalur jaklingko salah', 'muter-muter jaklingko', 'rute jaklingko salah',
        'akses halte jaklingko susah', 'jaklingko berenti di tempat salah', 'turun jaklingko di tempat salah',
        'naik jaklingko bingung', 'rute jaklingko gak jelas', 'jalur jaklingko gak nyambung',


    ],

    'Kemacetan': [
        'macet', 'parah banget', 'ngadat', 'padat merayap', 'terjebak macet', 'ngantri panjang',
        'berhenti total', 'jalan stuck', 'gak gerak', 'macet banget', 'macet parah',
        'terhambat karena macet', 'macet total', 'jalanan rame banget', 'stuck', 'ngantri parah',
        'penuh kendaraan', 'jalanan padat', 'total macet', 'galian fucek', 'antrian tj ke bekasi terlalu gilaaa',
        'stuck parah', 'ga gerak', 'jalanan penuh', 'macet total banget', 'berhenti di tengah jalan',
        'nunggu macet reda', 'macet panjang', 'macet di depan halte', 'macet di tol', 'macet di flyover',
        'berhenti lama karena macet','terjebak di jalan', 'macet parah banget', 'jalanan stuck banget', 'ga gerak lama',
        'berhenti lama di tengah jalan', 'macet total banget', 'macet terus tiap hari',
        'jalanan rame banget', 'macet panjang banget','kayak parkiran', 'macet total banget', 'macetnya gila', 'macet ampun',
        'jalannya stuck banget', 'macet beneran parah', 'ga gerak sedetik pun', 'macet mulu', 'sampe','jaklingko penuh',
        'gambling sama traffic','pake jaklingko dengan waktu 2 jam','satu jam jaklingko','maju 2 jam','jaklingko la andara saat ini lancar',
        'padat merayap', 'stuck', 'jalanan macet', 'terjebak macet', 'terhambat karena macet', 'berhenti di jalan',
        'macet parah', 'jalan terhambat', 'jalur macet', 'kendaraan padat', 'berhenti lama', 'berhenti macet', 'jalan stuck', 'macet banget', 'antrian panjang', 'jalan penuh kendaraan',
        'macet total', 'terjebak kemacetan', 'jalan lambat', 'macet total banget', 'terlalu macet','kejebak',
        'macet setiap hari', 'jalanan sempit', 'jalanan lambat', 'macet di tol', 'macet berjam-jam', 'macet parah banget',
        'stuck di tengah jalan', 'jalan penuh kendaraan', 'terlambat karena macet','1437 lalulintas jl s parman dari grogol ramai dan merayap jelang gt tanjung duren',
        '1405 jl suryopranoto dari harmoni ramai dan merayap mulai halte','kereta bogor udah lewat 4x tapi jakarta kota belum lewat',
        'macet', 'tersendat','jaklingko macet', 'macet di jaklingko', 'kemacetan di jaklingko', 'stuck di jaklingko', 'jalanan jaklingko padat',
        'ngantri jaklingko', 'macet di transjakarta', 'padat merayap jaklingko', 'macet total jaklingko','total diperjalanan 5 jam',
    ],

    'Ancaman': [
        'copet', 'kecurian', 'dompet ilang', 'hp hilang', 'rawan', 'takut', 'ngeri',
        'ancam', 'diganggu', 'pelecehan', 'ga aman', 'gak nyaman', 'dilecehkan', 'serem',
        'penjahat', 'preman', 'was-was', 'ada copet', 'kriminal', 'perampok', 'ancaman', 'nggak aman',
        'mabuk', 'pelecehan seksual', 'mesum', 'perampokan', 'diambil paksa', 'barang hilang',
        'copet profesional', 'cewek diganggu', 'salam tempel', 'dicolek', 'diraba', 'begal',
        'pencopetan', 'tertipu', 'perkelahian', 'keributan', 'berantem', 'pemalakan',
        'keamanan pelanggan adalah prioritas utama', 'tolong cekin bnrn satu2',
        'gelap banget', 'sepi banget', 'takut lewat sini', 'ada orang mabuk', 'ada yang ribut',
        'ada berantem', 'ngeri banget', 'bikin waswas', 'copet banyak', 'barang ilang',
        'dompet hilang', 'pencopetan', 'preman', 'gangguan penumpang', 'ribut di bus',
        'diserempet', 'dikejar', 'ngancem', 'bikin takut','gelap di halte', 'sepi banget', 'takut sendirian', 'ada orang mabuk',
        'ada yang ngikutin', 'ada ribut di bus', 'ngeri jalan malem', 'ada copet di bus',
        'ada yang ganggu penumpang', 'ada yang ngelihatin aneh',
        'sepi banget', 'takut banget', 'ngeri banget naik malam', 'ga aman banget',
        'gelap banget di halte', 'was was', 'deg-degan', 'bikin takut', 'nyeremin banget', 'ADA COPET',
        'mobilnya abis diserempet','ga berani sendiri','dilempar batu','pelemparan batu','aman kah','ilang','amaan',
        'halo totebag aku ketinggalan di kereta','menangkap pelaku penusukan di perlintasan rel kereta api',
        'maksudnya apa coba lempar batu pernah juga dulu naik kereta jakarta',
        'sering di lempar batu waktu naik kereta jakartabogor',
        'transum di jakarta sadar gak kalau iklan pinjol','rawan kejahatan', 'copet banyak', 'gak aman', 'takut naik sendirian', 'diikuti orang',
        'gangguan keamanan', 'begal', 'mabuk di kendaraan', 'diserempet', 'pencopetan', 'keributan penumpang',
        'barang hilang', 'kriminalitas tinggi', 'saat jalan malam takut', 'gak aman banget', 'diculik',
        'ancaman di perjalanan', 'pengamen mengganggu', 'ribut di dalam kendaraan', 'keributan saat perjalanan',
        'kereta yogyakarta surabaya dilempari batu penumpang', 'terhambat', 'gangguan', 'terkendala',
        'berbahaya', 'risiko tinggi', 'copet jaklingko', 'kriminalitas jaklingko', 'mabuk jaklingko', 'gangguan di jaklingko', 'preman jaklingko',
        'ancaman jaklingko', 'gangguan di bus jaklingko', 'pencopetan jaklingko', 'pengamen jaklingko',
        'bbrp thn yg lalu jg pernah ngalamin kena lemparan botol kratindaeng di kepala waktu naik krl bogor jakarta',

    ],

    'Infrastruktur': [
        'halte kotor', 'stasiun kumuh', 'eskalator mati', 'lift rusak', 'jalur sepi',
        'tempat nunggu', 'kursi di halte', 'fasilitas', 'toilet kotor', 'toilet rusak',
        'fasilitas minim', 'jembatan penyeberangan', 'atap bocor', 'berdebu', 'bau pesing',
        'tempat sampah penuh', 'kursi tunggu', 'petugas kebersihan', 'stasiun', 'halte tj',
        'jpo', 'loncat pager', 'berhenti di seberang gate yg ketutup pager', # BARU: Keluhan terkait JPO/akses fisik
        'halte jauh banget', 'halte ga ada', 'halte rusak', 'gate ketutup', 'akses susah',
        'jalan ke halte jauh', 'stasiun gelap', 'eskalator mati lagi', 'toilet bau', 'fasilitas rusak',
        'jembatan rusak', 'atap bocor', 'tempat duduk rusak', 'tempat nunggu kecil', 'halte becek',
        'halte ga layak', 'jpo licin', 'jalan kaki jauh ke halte','halte jauh banget dari sini', 'halte kecil banget',
        'halte ga layak', 'jpo licin','loncat pager', 'halte becek banget', 'jalan ke halte gelap', 'jpo susah dilewati',
        'toilet bau banget', 'stasiun kotor', 'jalan ke halte rusak','halte rusak', 'halte becek', 'jalan ke halte gelap', 'kursi tunggu rusak',
        'toilet rusak','tempat nunggu kecil', 'stasiun gelap', 'fasilitas rusak', 'kursi tidak nyaman', 'jembatan penyeberangan rusak',
        'toilet bau', 'pintu halte susah', 'pintu halte rusak', 'halte ga layak', 'escalator mati', 'lift rusak',
        'halte panas', 'halte sempit', 'halte kecil', 'halte becek banget', 'halte jelek',
        'halte rusak banget', 'ga ada tempat duduk', 'toilet bau banget', 'halte gelap',
        'halte kumuh', 'halte penuh', 'fasilitas minim banget','pintunya nyangkut','jalan jelek','acnya knp di jakbar gaada ya',
        'kondisi mobilnya','halte tidak berguna','kursi ekonominya gak tegak',
        'halte rusak', 'toilet bau', 'kursi tidak ada', 'stasiun penuh', 'fasilitas terbatas',
        'toilet kotor', 'jalur sempit', 'akses halte sulit', 'stasiun kotor', 'halte sempit',
        'eskalator mati', 'lift rusak', 'pintu halte susah dibuka', 'tempat duduk penuh', 'akses terbatas',
        'halte yang tidak layak', 'fasilitas yang buruk', 'jembatan penyeberangan rusak', 'kursi tunggu rusak',
        'halte jaklingko rusak', 'stasiun jaklingko gelap', 'fasilitas jaklingko minim', 'halte jaklingko sempit',
        'pintu halte jaklingko macet', 'jalan ke halte jaklingko jauh', 'toilet jaklingko bau', 'tempat duduk jaklingko penuh',
        'fasilitas halte jaklingko jelek', 'akses halte jaklingko susah','halo kapan ya lift dan eskalator cipulir diperbaiki',
        'antrian bus masuk cawang sentral tolong diurusin',

    ],

    'Akses/Rute': [
        'rute', 'jalur', 'naik bis yang mana', 'dari tmii ke csw', 'transmart cilandak arah moch kaffi',
        'turun di st jakot', 'turun di bus stop', 'koridor route 9', '9c bundaran senayan pinang ranti',
        'bisa naik jaklingko', 'masuk dan keluar', 'akses masuk', 'rute gw', 'lewat rute gw', 'nyari penumpang',
        'ga dari bus stop', 'jaklingko berenti beroperasi dr jam 18', 'berenti beroperasi', # BARU: Isu jam operasional
        'gada jaklingko yang berhenti langsung depan unj', 'jalur',
        'berhenti beroperasi jam', 'sudah tutup jam', 'ga lewat sini lagi', 'jalur berubah',
        'rute ditutup', 'akses susah', 'naik dari mana', 'turun di mana', 'ga ada jalurnya',
        'akses ke halte susah', 'jalur ga nyambung', 'rute potong', 'ga dari halte resmi',
        'berhenti di luar halte', 'rute ditutup sore','ga lewat sini lagi', 'rute berubah tiba-tiba', 'berhenti di tempat lain',
        'jalur dipotong', 'akses susah ke halte', 'ga nyambung sama transjakarta',
        'jalur ke kokas ribet', 'ga ada arah ke sana', 'bus berhenti bukan di halte',
        'ga nyambung sama koridor', 'ganti bus dua kali',
        'jalur dipotong', 'rute berubah tiba‑tiba', 'akses ke halte susah', 'ganti bus dua kali',
        'naik dari mana', 'bus berhenti bukan di halte', 'lewat halte', 'bablas halte', 'berhenti jauh dari halte',
        'rute ngaco', 'rute yang aneh', 'rute yang bingung', 'akses susah ke halte', 'jalur tidak jelas',
        'jalur berubah', 'jalur dipotong', 'rute ganti', 'rute berubah tiba tiba',
        'ga nyambung sama TJ', 'akses ribet banget', 'rute aneh banget','jaklingko bisa muter jauhh',
        'gak ada 6h yang berhenti disitu semuanya lewat tol ternyata',
        'rute salah', 'akses sulit', 'rute tidak terhubung', 'berhenti di luar halte', 'jalur tidak jelas',
        'akses ke halte sulit', 'rute potong', 'jalur tidak sampai', 'rute ga nyambung', 'jalur berubah',
        'salah naik', 'naik di halte tidak resmi', 'perjalanan memutar', 'akses ke bus berhenti jauh',
        'gangguan jalan', 'akses susah', 'rute ngaco', 'terbatas aksesnya','kesulitan mengakses',
        'rute jaklingko tutup', 'akses jaklingko susah', 'rute jaklingko berubah', 'berhenti jaklingko di luar halte',
        'rute jaklingko gak nyambung', 'akses halte jaklingko ribet', 'perjalanan jaklingko muter-muter',

    ],

    'Prioritas/Kebutuhan Khusus': [
        'difabel', 'disabilitas', 'ramah disabilitas', 'kursi roda', 'akses sulit',
        'ibu hamil', 'lansia', 'tempat duduk prioritas', 'dipersulit', 'butuh bantuan',
        'tidak ada ramp', 'tangga', 'duduk di depan', 'bilang lg hamil',
        'susah naik', 'susah turun', 'ga ada ramp', 'ga bisa kursi roda', 'ga dikasih duduk',
        'duduk prioritas diambil', 'ga ada tempat duduk prioritas', 'ga ramah disabilitas',
        'susah buat lansia', 'ibu hamil berdiri', 'ga dikasih tempat duduk',
        'ga dikasih duduk padahal hamil', 'kursi prioritas diambil', 'lansia berdiri',
        'difabel susah naik', 'kursi roda susah masuk', 'ga ramah disabilitas',
        'kasian ibu hamil', 'lansia berdiri', 'ga dikasih duduk', 'duduk prioritas diambil',
        'ga dikasih kursi', 'ibu hamil ga dapet tempat duduk', 'kursi prioritas dipakai',
        'ramah disabilitas', 'kursi prioritas diambil', 'difabel susah naik', 'ibu hamil ga dikasih duduk',
        'lansia berdiri', 'difabel gak dilayani', 'akses susah untuk difabel',
        'kursi roda sulit masuk', 'akses terbatas bagi disabilitas', 'susah buat ibu hamil',
        'kursi prioritas jaklingko diambil', 'akses jaklingko untuk ibu hamil', 'difabel jaklingko susah naik',
        'kursi prioritas jaklingko penuh', 'akses jaklingko untuk lansia terbatas',

    ],

    'Emosi/Frustrasi': [
        'bikin emosi', 'nyesel naik jaklingko hari ini', 'ter sial', 'anyink', 'anjir', 'sial', 'kesel banget',
        'wkwk random amat', 'crying in jaklingko', 'kena passinghae jaklingko kena bgt di gue', 'kesel', 'bete',
        'cape banget', 'males banget', 'anjay', 'ampun dah', 'cape deh', 'nyesel banget', 'stress', 'gila sih', 'kzl',
        'pusing', 'cape hati', 'emosi banget', 'ga kuat lagi', 'sumpah kesel', 'cape nunggu', 'bosen', 'muak', 'cape banget nunggu',
        'kesel banget sama jaklingko', 'nyesel naik jaklingko', 'capek banget', 'bete banget', 'anjrit kesel', 'kesel parah',
        'cape batin', 'muakk banget', 'pusing banget', 'bikin stres', 'kesel banget', 'stress banget', 'cape deh', 'capek banget',
        'terlalu lama', 'kesel parah', 'stress banget', 'bete banget', 'emosi banget', 'sabar abis', 'bikin kesel',
        'kesel banget dah', 'bete dah', 'males banget naik ini', 'kesel banget sama sistemnya', 'cape hati banget',
        'muak banget', 'stress parah', 'SUMPAH ANJIIRRR', 'benci', 'mual gua naek tj', 'males bgt', 'ketidaknyamanannya',
        'ah kontoooollll', 'malah gesrek', 'gilaaaa', 'bngsat', 'goblok', 'ketinggalan kereta', 'ajg', 'mabok jaklingko',
        'cape bgt', 'tolol', 'kampret', 'bgst', 'mellow', 'pertama kali', 'kayak roda', 'jaklingko tidur',
        'jalanan umum buat angkutan umum bukan buat hajatan','bisa jaklingko ya thank you', 'lelah bgt', 'malah ngedumel','dimarahin', 'gengsi', 'gemez banget', 'berisik','ngaco bgt',
        'duduk sebelahan sama masmas di jaklingko gua kaget garagara masnya turun nunduk bajunya keangkat dan ya cd thongnya keliatan dan pas muka gua ituu',
        'nangis', 'cape', 'random bgt','kocak', 'buset', 'gak nyambung', 'apa susahnya niru','masih byk anak2 di daerah bodetabek',
        'kesenjangan yg ak alamai', 'jaklingko emng seugal an', 'lelah', 'sangat2 buruk','jalanan umum buat angkutan umum bukan buat hajatan',
        'alasan mulu', 'badan pegel', 'kaget bjir', 'ketidaknyamanan yang dialami', 'ku tahan biar ga tidur','biasanya dari jakarta pulang naik kereta turun di semarang tawang',
        'mana progressnya masih sama keadaannya', 'enak bgt', 'terkutuk ini koridor', 'enak jg', 'ayout bis tj paling aneh',
        'keparat2 itu', 'nyeseq pisun', 'krl arah jakarta haeusnya','gausa pake jaklingko lg daaaah','di jaklingko gugup','anying',
        'curiga noh koridor kena santet','tiket jakarta 1 kali kereta bandara bangkok pp miris hati','rasa was-was','kalo naik jaklingko itu kakinya rapetin',
        'takut', 'tidak nyaman', 'kekhawatiran', 'kesulitan','kesel naik jaklingko', 'stress naik jaklingko', 'bete banget sama jaklingko', 'muak naik jaklingko',
        'nyesel naik jaklingko', 'males banget naik jaklingko', 'cape banget nunggu jaklingko', 'kesel banget sama sistem jaklingko',' trauma diomelin','koridor 9',

  ],

    'Regulasi/Operasional': [ # KATEGORI BARU UNTUK ATURAN/SISTEM
        'ngide', 'gak mempan', 'sok atuh soan belajar ka pak anies', 'cuma versi belom gratisnya',
        'siap menambah skillset baru tj jaklingko', # Untuk kasus yang tidak jelas
        'berhenti operasi', 'ga beroperasi', 'tutup jam segini', 'jadwal berubah',
        'aturan baru', 'sistem baru', 'ga dikasih info', 'aturan aneh', 'operasional buruk',
        'jadwal ga sesuai', 'tiba-tiba ditutup', 'layanan ditutup', 'pengumuman dadakan'
        'operasional ngaco', 'jadwal tiba-tiba berubah', 'ga dikasih info jam operasi',
        'berhenti operasi jam 6', 'layanan berhenti mendadak', 'sistem berubah',
        'aturan aneh banget', 'ga jelas jam operasional', 'pemberitahuan mendadak',
        'jadwal ga tentu', 'tiba tiba tutup', 'berhenti tiba tiba', 'sistem aneh',
        'layanan ga konsisten', 'tiba tiba berubah', 'operasional ngaco banget','tiket di app jaklingko bisa buat tj nonbrt',
        'asal sistemnya jelas','lu setuju gak jaklingko bisa dipesen online',
        'ya makanya jgn dihapus ganti aja pake sistem gaji kek jaklingko',
        'sh1 dibanyakin armadanya masa 30 menit sekali','contoh aja jaklingko angkot tetap beroperasi tapi trayek',
        'cuma 1 akses buat turun ke ariobimopalmamega syariah','pagernya harusnya dibuka dan kasi zebra cross buat pintu',
        'sterilin aja dari kendaraan sisain 1 lajur buat tj amp jaklingko','biasanya gini chri bikin wacanaprogram angkot modern kyk jaklingko baru',
        'berhenti operasi', 'ga beroperasi', 'jadwal berubah', 'layanan ditutup',
        'operasional ngaco', 'layanan tidak konsisten', 'pengumuman mendadak',
        'pagernya harusnya dibuka dan kasi zebra cross buat pintu yg di depan halte karena gua ngerasain naik tjjaklingko',
        'jadwal tidak jelas', 'aturan aneh', 'jadwal berubah mendadak', 'pengumuman tidak ada',
        'layanan tutup lebih awal', 'gagal sistem operasional', 'operasional yang tidak terduga',
        'kejadian lagi nih pagi ini masih belum si evaluasi juga nih jadwal bus',
        'sering sekali kendaraan menabrak mcb knp tidak dicari solusi misal pasang rambu lampu yg jelas',
        'menurut saya sih angkot trayek dibandingkan dengan alternatif ondemand dan penghapusan total',
        'aturan jaklingko baru', 'operasional jaklingko berubah', 'jadwal jaklingko tiba-tiba berubah', 'sistem jaklingko baru',
        'pengumuman mendadak jaklingko', 'layanan jaklingko tutup', 'berhenti operasi jaklingko', 'operasional jaklingko kacau',
        'harusnya jg jaklingko app sedia opsi kek gini lebih bagus','butuh tj metrotrans dan jaklingko di bekas',
        'sayangnya jakarta tuh klo mo terkoneksi dg transum tuh harus ke trngah dulu','seharusnya sih angkot2 itu bisa di integrasikan ke sistem'
        'polri dan ypktb siapkan pemimpin masa depan lewat kereta kader','realistis maksimalkan transpatriot kalo emang gabisa',
    ]
    }
    
    detected_problems = []
    text_lower = text.lower()
    
    for problem, keywords in problem_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_problems.append(problem)
                break
    
    return list(set(detected_problems))

# CSS styling yang adaptif untuk dark/light mode
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: var(--secondary-color);
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .tweet-card {
        background-color: var(--background-color);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        color: var(--text-color);
        border: 1px solid var(--border-color);
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
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
        border: 1px solid var(--border-color);
    }
    .metric-card {
        background-color: var(--card-background-color);
        color: var(--card-text-color);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
        border: 1px solid var(--border-color);
    }
    .metric-card h3 {
        color: var(--card-text-color) !important;
        margin-bottom: 10px;
        font-size: 1rem;
    }
    .metric-card h2 {
        color: var(--card-text-color) !important;
        margin: 0;
        font-size: 1.8rem;
    }
    .metric-card p {
        color: var(--card-text-color) !important;
        margin: 5px 0 0 0;
    }
    .tweet-text {
        margin: 0 0 10px 0;
        font-size: 0.95rem;
        line-height: 1.4;
        color: var(--text-color);
    }
    .tweet-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }
    .problems-container {
        margin-top: 8px;
    }
    
    /* CSS Variables untuk Dark/Light Mode */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #2e86ab;
        --background-color: #f8f9fa;
        --card-background-color: #ffffff;
        --secondary-background-color: #e9ecef;
        --text-color: #000000;
        --card-text-color: #000000;
        --border-color: #dee2e6;
    }
    
    /* Dark Mode Styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #4da8ff;
            --secondary-color: #6cb8ff;
            --background-color: #1e1e1e;
            --card-background-color: #2d2d2d;
            --secondary-background-color: #3d3d3d;
            --text-color: #ffffff;
            --card-text-color: #ffffff;
            --border-color: #404040;
        }
    }
    
    /* Streamlit Dark Mode Compatibility */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Memastikan text di sidebar juga adaptif */
    .css-1d391kg, .css-12oz5g7, .css-1y4p8pa {
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<div class="main-header">🚍 Dashboard Analisis Sentimen Transportasi Jakarta</div>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_df.csv")
        st.success("✅ Data berhasil dimuat dari final_df.csv")
        return df
    except Exception as e:
        st.warning(f"⚠️ Tidak dapat memuat final_df.csv: {e}. Menggunakan data sample...")
        # Fallback data sample
        data = {
            'Kategori': ['jak', 'jak', 'tj', 'tj', 'krl', 'krl', 'jak', 'tj'],
            'Tweet': [
                'Jaklingko sangat nyaman dan tepat waktu hari ini',
                'Saya menunggu Jaklingko terlalu lama, keterlambatan yang menyebalkan',
                'Transjakarta AC-nya dingin dan sopirnya ramah',
                'Bus Transjakarta penuh sesak dan tidak nyaman',
                'KRL hari ini berjalan dengan lancar dan nyaman',
                'KRL sangat padat dan berisik, tidak nyaman',
                'Jaklingko gratis membuat pengeluaran bulanan lebih hemat',
                'Rute Transjakarta semakin lengkap dan terintegrasi'
            ],
            'Sentiment': ['Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Negatif', 'Positif', 'Positif'],
            'problem': [
                "['Kenyamanan']",
                "['Keterlambatan', 'Emosi/Frustrasi']",
                "['Kenyamanan', 'Pelayanan']",
                "['Kondisi', 'Kenyamanan']",
                "['Kenyamanan']",
                "['Kondisi', 'Kenyamanan']",
                "['Harga']",
                "['Akses/Rute']"
            ]
        }
        df = pd.DataFrame(data)
        return df

df = load_data()

# Preprocess problem data - FIXED VERSION
def preprocess_problems(problem_str):
    if pd.isna(problem_str) or problem_str == '[]' or problem_str == '':
        return []
    
    try:
        # Coba parsing sebagai Python list
        if isinstance(problem_str, str) and problem_str.startswith('['):
            try:
                problems = ast.literal_eval(problem_str)
                if isinstance(problems, list):
                    return [str(p).strip() for p in problems if p and str(p).strip()]
            except:
                pass
        
        # Fallback: manual parsing
        if isinstance(problem_str, str):
            # Remove brackets and quotes
            clean_str = problem_str.strip("[]'\" ")
            if clean_str:
                # Split by comma and clean
                problems = [p.strip().strip("'\"") for p in clean_str.split(',')]
                return [p for p in problems if p]
        
        return []
    except Exception as e:
        return []

# Apply preprocessing
df['problems_clean'] = df['problem'].apply(preprocess_problems)

# Debug info
st.sidebar.markdown(f"**Dataset Info:** {len(df)} baris data dimuat")

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
    
    analyze_btn = st.button("Analisis Komentar", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.markdown("""
    Dashboard ini menganalisis sentimen dan masalah pada transportasi Jakarta:
    - **JakLingko**: Mikrotrans terintegrasi
    - **TransJakarta**: Bus Rapid Transit  
    - **KRL**: Kereta Rel Listrik
    """)

# Global variable untuk menyimpan data baru
if 'new_comments' not in st.session_state:
    st.session_state.new_comments = []

# Proses analisis komentar baru
if analyze_btn and new_comment:
    with st.spinner("Menganalisis sentimen dan masalah..."):
        # Analisis sentimen
        sentiment, confidence = analyze_sentiment(new_comment, tokenizer, model)
        
        # Deteksi masalah
        problems = detect_problems(new_comment)
        
        # Tampilkan hasil
        st.success("✅ Analisis selesai!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_color = {
                "Positif": "sentiment-positive",
                "Negatif": "sentiment-negative", 
                "Netral": "sentiment-neutral"
            }
            st.markdown(f'''
            <div class="metric-card">
                <h3>Sentimen</h3>
                <p class="{sentiment_color[sentiment]}">{sentiment}</p>
                <p>Confidence: {confidence:.2f}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>Kategori</h3>
                <p>{"JakLingko" if transport_category == "jak" else "TransJakarta" if transport_category == "tj" else "KRL"}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            problems_text = ", ".join(problems) if problems else "Tidak terdeteksi"
            st.markdown(f'''
            <div class="metric-card">
                <h3>Masalah Terdeteksi</h3>
                <p>{problems_text}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Simpan ke session state
        new_comment_data = {
            'Kategori': transport_category,
            'Tweet': new_comment,
            'Sentiment': sentiment,
            'problem': str(problems),
            'problems_clean': problems
        }
        st.session_state.new_comments.append(new_comment_data)
        
        st.info("📝 Data telah ditambahkan. Visualisasi akan diperbarui.")

# Gabungkan data baru dengan data utama
if st.session_state.new_comments:
    new_df = pd.DataFrame(st.session_state.new_comments)
    combined_df = pd.concat([df, new_df], ignore_index=True)
else:
    combined_df = df.copy()

# Tabs untuk masing-masing transportasi
tab1, tab2, tab3 = st.tabs(["🚐 JakLingko", "🚍 TransJakarta", "🚆 KRL"])

def create_transport_tab(category, category_name):
    # Filter data berdasarkan kategori
    category_data = combined_df[combined_df['Kategori'] == category].copy()
    
    if category_data.empty:
        st.warning(f"📭 Tidak ada data untuk {category_name}")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tweets = len(category_data)
    positive_tweets = len(category_data[category_data['Sentiment'] == 'Positif'])
    negative_tweets = len(category_data[category_data['Sentiment'] == 'Negatif'])
    neutral_tweets = len(category_data[category_data['Sentiment'] == 'Netral'])
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Total Tweet</h3>
            <h2>{total_tweets}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Positif</h3>
            <h2 style="color: #28a745;">{positive_tweets}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Negatif</h3>
            <h2 style="color: #dc3545;">{negative_tweets}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Netral</h3>
            <h2 style="color: #ffc107;">{neutral_tweets}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Visualisasi 1: Top Problems
    st.markdown(f'<div class="sub-header">📊 Top 5 Masalah pada {category_name}</div>', unsafe_allow_html=True)
    
    # Extract all problems
    all_problems = []
    for problems in category_data['problems_clean']:
        if problems:  # Only extend if problems is not empty
            all_problems.extend(problems)
    
    if all_problems:
        problem_counts = Counter(all_problems)
        top_problems = problem_counts.most_common(5)
        
        if top_problems:
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
            fig.update_layout(
                showlegend=False, 
                yaxis={'categoryorder':'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='var(--text-color)')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Tidak ada masalah yang terdeteksi dalam data")
    else:
        st.info("ℹ️ Belum ada data masalah yang terdeteksi")
    
    # Visualisasi 2: Layout columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'<div class="sub-header">📈 Distribusi Sentimen</div>', unsafe_allow_html=True)
        
        sentiment_counts = category_data['Sentiment'].value_counts()
        if not sentiment_counts.empty:
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
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='var(--text-color)')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("ℹ️ Tidak ada data sentimen")
    
    with col2:
        st.markdown(f'<div class="sub-header">🔍 Tren Masalah</div>', unsafe_allow_html=True)
        
        if all_problems:
            # Problem frequency table
            problem_freq = pd.DataFrame(problem_counts.most_common(10), columns=['Problem', 'Frekuensi'])
            st.dataframe(problem_freq, use_container_width=True, height=300)
        else:
            st.info("ℹ️ Belum ada data masalah")
    
    # Tweet terbaru - FIXED DISPLAY
        # Tweet terbaru - Hanya tampilkan yang memiliki problem
    st.markdown(f'<div class="sub-header">💬 Tweet Terbaru tentang {category_name}</div>', unsafe_allow_html=True)
    
    # Filter hanya tweet yang memiliki problem
    tweets_with_problems = category_data[category_data['problems_clean'].apply(lambda x: len(x) > 0)]
    
    if not tweets_with_problems.empty:
        recent_tweets = tweets_with_problems.tail(8).iloc[::-1]  # Reverse untuk dapat yang terbaru di atas
        
        for _, tweet in recent_tweets.iterrows():
            sentiment_class = f"sentiment-{tweet['Sentiment'].lower()}"
            
            # Handle problems display safely
            problems_html = ""
            if tweet['problems_clean'] and len(tweet['problems_clean']) > 0:
                problems_html = "<div class='problems-container'>"
                for problem in tweet['problems_clean']:
                    if problem and str(problem).strip():  # Only add if problem is not empty
                        problems_html += f'<span class="problem-tag">{problem}</span>'
                problems_html += "</div>"
            
            # Highlight new comments
            is_new = any(tweet['Tweet'] == nc['Tweet'] for nc in st.session_state.new_comments)
            border_color = "#ff6b6b" if is_new else "var(--primary-color)"
            
            # Fixed HTML structure
            tweet_html = f"""
            <div class="tweet-card" style="border-left-color: {border_color};">
                <p class="tweet-text">{tweet['Tweet']}</p>
                <div class="tweet-footer">
                    <div class="problems-container">
                        {problems_html}
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
    "<div style='color: var(--text-color);'>"
    "<strong>Dashboard Analisis Sentimen Transportasi Jakarta</strong> | "
    "Data diperbarui secara real-time dengan analisis AI | "
    f"Total data: {len(combined_df)} komentar"
    "</div>", 
    unsafe_allow_html=True
)

# Tombol reset data baru
if st.session_state.new_comments:
    if st.button("🔄 Reset Data Baru", type="secondary"):
        st.session_state.new_comments = []
        st.rerun()






