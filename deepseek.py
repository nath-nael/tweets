import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Dashboard Analisis Transportasi Jakarta",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .tweet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        transition: transform 0.3s ease;
    }
    
    .tweet-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .tweet-text {
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .tweet-meta {
        font-size: 0.85rem;
        opacity: 0.9;
        font-style: italic;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E40AF;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }
    
    .input-section {
        background: #F9FAFB;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #E5E7EB;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def load_model():
    model_name = "w11wo/indobert-large-p1-twitter-indonesia-sarcastic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Problem keywords from documen
PROBLEM_KEYWORDS= {
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

# Analyze sentiment
def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_idx = torch.argmax(probs).item()
    
    sentiments = ['negatif', 'netral', 'positif']
    return sentiments[sentiment_idx]

# Detect problems
def detect_problems(text):
    text_lower = text.lower()
    detected = []
    
    for problem, keywords in PROBLEM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(problem)
                break
    
    return detected if detected else ['Lainnya']

# Load data
@st.cache_data
def load_data():
    # Create sample data if file doesn't exist
    try:
        df = pd.read_csv('final_df.csv')
    except:
        df = pd.DataFrame({
            'Tweet': [
                'TransJakarta hari ini telat banget, nunggu 1 jam!',
                'JakLingko nyaman dan bersih',
                'KRL penuh sesak, AC mati lagi',
                'Pelayanan sopir TJ kasar',
                'JakLingko tepat waktu, mantap!'
            ],
            'Kategori': ['tj', 'jak', 'krl', 'tj', 'jak'],
            'Sentiment': ['negatif', 'positif', 'negatif', 'negatif', 'positif'],
            'problem': ['Keterlambatan', 'Lainnya', 'Kondisi', 'Pelayanan', 'Lainnya']
        })
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'new_comments' not in st.session_state:
    st.session_state.new_comments = []

# Load model
tokenizer, model = load_model()

# Header
st.markdown('<h1 class="main-header">🚌 Dashboard Analisis Transportasi Jakarta</h1>', unsafe_allow_html=True)

# Sidebar stats
with st.sidebar:
    st.header("📊 Statistik Keseluruhan")
    
    total_tweets = len(st.session_state.df)
    positive = len(st.session_state.df[st.session_state.df['Sentiment'] == 'positif'])
    negative = len(st.session_state.df[st.session_state.df['Sentiment'] == 'negatif'])
    neutral = len(st.session_state.df[st.session_state.df['Sentiment'] == 'netral'])
    
    st.metric("Total Tweet", total_tweets)
    st.metric("Positif", positive, delta=f"{positive/total_tweets*100:.1f}%")
    st.metric("Negatif", negative, delta=f"-{negative/total_tweets*100:.1f}%", delta_color="inverse")
    st.metric("Netral", neutral, delta=f"{neutral/total_tweets*100:.1f}%")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🚌 TransJakarta", "🚐 JakLingko", "🚊 KRL"])

def create_tab_content(kategori, name):
    df_filtered = st.session_state.df[st.session_state.df['Kategori'] == kategori]
    
    if len(df_filtered) == 0:
        st.warning(f"Tidak ada data untuk {name}")
        return
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{len(df_filtered)}</div><div class="stat-label">Total Tweet</div></div>', unsafe_allow_html=True)
    with col2:
        pos_pct = len(df_filtered[df_filtered['Sentiment'] == 'positif']) / len(df_filtered) * 100
        st.markdown(f'<div class="stat-card"><div class="stat-value">{pos_pct:.1f}%</div><div class="stat-label">Sentimen Positif</div></div>', unsafe_allow_html=True)
    with col3:
        neg_pct = len(df_filtered[df_filtered['Sentiment'] == 'negatif']) / len(df_filtered) * 100
        st.markdown(f'<div class="stat-card"><div class="stat-value">{neg_pct:.1f}%</div><div class="stat-label">Sentimen Negatif</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top 5 Problems
    st.subheader(f"🔥 Top 5 Masalah {name}")
    
    problem_counts = df_filtered['problem'].value_counts().head(5)
    
    fig = go.Figure(data=[
        go.Bar(
            x=problem_counts.values,
            y=problem_counts.index,
            orientation='h',
            marker=dict(
                color=problem_counts.values,
                colorscale='Reds',
                showscale=True
            ),
            text=problem_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Top 5 Masalah pada {name}",
        xaxis_title="Jumlah Keluhan",
        yaxis_title="Kategori Masalah",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_counts = df_filtered['Sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribusi Sentimen",
            color=sentiment_counts.index,
            color_discrete_map={
                'positif': '#10B981',
                'negatif': '#EF4444',
                'netral': '#6B7280'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        problem_by_sentiment = df_filtered.groupby(['problem', 'Sentiment']).size().reset_index(name='count')
        fig_bar = px.bar(
            problem_by_sentiment,
            x='problem',
            y='count',
            color='Sentiment',
            title="Masalah berdasarkan Sentimen",
            color_discrete_map={
                'positif': '#10B981',
                'negatif': '#EF4444',
                'netral': '#6B7280'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent tweets
    st.subheader(f"💬 Tweet Terbaru {name}")
    
    recent_tweets = df_filtered.tail(10).iloc[::-1]
    
    for idx, row in recent_tweets.iterrows():
        sentiment_class = f"sentiment-{row['Sentiment']}"
        st.markdown(f"""
        <div class="tweet-card {sentiment_class}">
            <div class="tweet-text">{row['Tweet']}</div>
            <div class="tweet-meta">
                Sentimen: <strong>{row['Sentiment'].upper()}</strong> | 
                Problem: <strong>{row['problem']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab1:
    create_tab_content('tj', 'TransJakarta')

with tab2:
    create_tab_content('jak', 'JakLingko')

with tab3:
    create_tab_content('krl', 'KRL')

# Input section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("✍️ Analisis Komentar Baru")

col1, col2 = st.columns([3, 1])

with col1:
    user_comment = st.text_area(
        "Masukkan komentar Anda tentang transportasi umum Jakarta:",
        height=100,
        placeholder="Contoh: TransJakarta hari ini telat 30 menit, capek nunggunya..."
    )

with col2:
    kategori_input = st.selectbox(
        "Pilih Moda:",
        options=['tj', 'jak', 'krl'],
        format_func=lambda x: {'tj': 'TransJakarta', 'jak': 'JakLingko', 'krl': 'KRL'}[x]
    )

if st.button("🔍 Analisis Komentar", type="primary", use_container_width=True):
    if user_comment.strip():
        with st.spinner("Menganalisis komentar..."):
            # Analyze sentiment
            sentiment = analyze_sentiment(user_comment, tokenizer, model)
            
            # Detect problems
            problems = detect_problems(user_comment)
            problem = problems[0] if problems else 'Lainnya'
            
            # Add to dataframe
            new_row = pd.DataFrame({
                'Tweet': [user_comment],
                'Kategori': [kategori_input],
                'Sentiment': [sentiment],
                'problem': [problem]
            })
            
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            
            # Display results
            st.success("✅ Analisis selesai!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentimen", sentiment.upper())
            with col2:
                st.metric("Problem Terdeteksi", problem)
            with col3:
                st.metric("Kategori", kategori_input.upper())
            
            st.info("💡 Dashboard telah diperbarui dengan data baru. Silakan cek tab yang sesuai!")
            st.balloons()
    else:
        st.warning("⚠️ Mohon masukkan komentar terlebih dahulu.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 2rem;'>
    <p>Dashboard Analisis Transportasi Jakarta | Dibuat dengan ❤️ menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)
