# %% [markdown]
# # **1. Perkenalan Dataset**
# 

# %% [markdown]
# Tahap pertama, Anda harus mencari dan menggunakan dataset **tanpa label** dengan ketentuan sebagai berikut:
# 
# 1. **Sumber Dataset**:  
#    Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (*Kaggle*, *UCI ML Repository*, *Open Data*) atau data primer yang Anda kumpulkan sendiri.
#    
# 2. **Ketentuan Dataset**:
#    - **Tanpa label**: Dataset tidak boleh memiliki label atau kelas.
#    - **Jumlah Baris**: Minimal 1000 baris untuk memastikan dataset cukup besar untuk analisis yang bermakna.
#    - **Tipe Data**: Harus mengandung data **kategorikal** dan **numerikal**.
#      - *Kategorikal*: Misalnya jenis kelamin, kategori produk.
#      - *Numerikal*: Misalnya usia, pendapatan, harga.
# 
# 3. **Pembatasan**:  
#    Dataset yang sudah digunakan dalam latihan clustering (seperti customer segmentation) tidak boleh digunakan.

# %% [markdown]
# # **2. Import Library**

# %% [markdown]
# Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # **3. Memuat Dataset**

# %% [markdown]
# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
# 
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.

# %%
# Set style untuk visualisasi
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Membaca dataset dari file CSV
df = pd.read_csv('bank_transactions_data_2.csv')

# Menampilkan 5 baris pertama dari dataset
print("5 Baris Pertama Dataset:")
print(df.head())

# %% [markdown]
# # **4. Exploratory Data Analysis (EDA)**
# 
# Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset. EDA bertujuan untuk:
# 
# 1. **Memahami Struktur Data**
#    - Tinjau jumlah baris dan kolom dalam dataset.  
#    - Tinjau jenis data di setiap kolom (numerikal atau kategorikal).
# 
# 2. **Menangani Data yang Hilang**  
#    - Identifikasi dan analisis data yang hilang (*missing values*). Tentukan langkah-langkah yang diperlukan untuk menangani data yang hilang, seperti pengisian atau penghapusan data tersebut.
# 
# 3. **Analisis Distribusi dan Korelasi**  
#    - Analisis distribusi variabel numerik dengan statistik deskriptif dan visualisasi seperti histogram atau boxplot.  
#    - Periksa hubungan antara variabel menggunakan matriks korelasi atau scatter plot.
# 
# 4. **Visualisasi Data**  
#    - Buat visualisasi dasar seperti grafik distribusi dan diagram batang untuk variabel kategorikal.  
#    - Gunakan heatmap atau pairplot untuk menganalisis korelasi antar variabel.
# 
# Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.

# %%
# Informasi struktur data
print("\nInformasi Dataset:")
print(df.info())

# Statistik deskriptif untuk kolom numerik
print("\nStatistik Deskriptif Kolom Numerik:")
print(df.describe())

# Memeriksa nilai yang hilang
print("\nJumlah Nilai yang Hilang untuk Setiap Kolom:")
print(df.isnull().sum())

# Distribusi data kategorikal
print("\nDistribusi Nilai untuk Kolom Kategorikal:")
categorical_columns = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
for col in categorical_columns:
    print(f"\nDistribusi {col}:")
    print(df[col].value_counts())
    
    # Visualisasi untuk distribusi kategorikal
    plt.figure(figsize=(10, 6))
    if col == 'Location':  # Untuk Location, kita gunakan plot horizontal karena banyak nilai unik
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribusi {col}')
        plt.tight_layout()
    else:
        sns.countplot(x=col, data=df)
        plt.title(f'Distribusi {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.show()

# Distribusi data numerik
numerical_columns = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 
                     'LoginAttempts', 'AccountBalance']
for col in numerical_columns:
    print(f"\nStatistik untuk {col}:")
    print(df[col].describe())
    
    # Visualisasi untuk distribusi numerik
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribusi {col}')
    plt.tight_layout()
    plt.show()

# Matriks korelasi untuk kolom numerik
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriks Korelasi')
plt.tight_layout()
plt.show()

# %% [markdown]
# # **5. Data Preprocessing**

# %% [markdown]
# Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.
# 
# Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi **tidak terbatas** pada:
# 1. Menghapus atau Menangani Data Kosong (Missing Values)
# 2. Menghapus Data Duplikat
# 3. Normalisasi atau Standarisasi Fitur
# 4. Deteksi dan Penanganan Outlier
# 5. Encoding Data Kategorikal
# 6. Binning (Pengelompokan Data)
# 
# Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah.

# %%
# Menyeleksi fitur yang akan digunakan untuk clustering
features = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 
           'LoginAttempts', 'AccountBalance', 'TransactionType', 
           'Channel', 'CustomerOccupation']

# Memisahkan fitur numerik dan kategorikal
numerical_features = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 
                      'LoginAttempts', 'AccountBalance']
categorical_features = ['TransactionType', 'Channel', 'CustomerOccupation']

# Membuat subset data dengan fitur yang diseleksi
X = df[features].copy()

# Preprocessor untuk standardisasi dan encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Fit dan Transform data
X_processed = preprocessor.fit_transform(X)

# %% [markdown]
# # **6. Pembangunan Model Clustering**

# %% [markdown]
# ## **a. Pembangunan Model Clustering**

# %% [markdown]
# Pada tahap ini, Anda membangun model clustering dengan memilih algoritma yang sesuai untuk mengelompokkan data berdasarkan kesamaan. Berikut adalah **rekomendasi** tahapannya.
# 1. Pilih algoritma clustering yang sesuai.
# 2. Latih model dengan data menggunakan algoritma tersebut.

# %%
# Menggunakan Elbow method untuk menentukan jumlah cluster yang optimal
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    # Inisialisasi dan fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    
    # Menghitung inertia (untuk Elbow method)
    inertia.append(kmeans.inertia_)
    
    # Menghitung Silhouette Score
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_processed, labels))
    
    print(f"k={k}, inertia={kmeans.inertia_:.2f}, silhouette score={silhouette_score(X_processed, labels):.4f}")

# Visualisasi Elbow method
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.grid(True)

# Visualisasi Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.title('Silhouette Score Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Berdasarkan hasil analisis, silhouette score tertinggi pada k=5
optimal_clusters = 5

# Membangun model KMeans dengan jumlah cluster optimal
kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
clusters = kmeans_optimal.fit_predict(X_processed)

# Menambahkan label cluster ke DataFrame
df['Cluster'] = clusters

# %% [markdown]
# ## **b. Evaluasi Model Clustering**

# %% [markdown]
# Untuk menentukan jumlah cluster yang optimal dalam model clustering, Anda dapat menggunakan metode Elbow atau Silhouette Score.
# 
# Metode ini membantu kita menemukan jumlah cluster yang memberikan pemisahan terbaik antar kelompok data, sehingga model yang dibangun dapat lebih efektif. Berikut adalah **rekomendasi** tahapannya.
# 1. Gunakan Silhouette Score dan Elbow Method untuk menentukan jumlah cluster optimal.
# 2. Hitung Silhouette Score sebagai ukuran kualitas cluster.

# %%
sil_score = silhouette_score(X_processed, clusters)
print(f"\nSilhouette Score untuk model dengan {optimal_clusters} cluster: {sil_score:.4f}")

# %% [markdown]
# ## **c. Feature Selection (Opsional)**

# %% [markdown]
# Silakan lakukan feature selection jika Anda membutuhkan optimasi model clustering. Jika Anda menerapkan proses ini, silakan lakukan pemodelan dan evaluasi kembali menggunakan kolom-kolom hasil feature selection. Terakhir, bandingkan hasil performa model sebelum dan sesudah menerapkan feature selection.

# %%
#Type your code here

# %% [markdown]
# ## **d. Visualisasi Hasil Clustering**

# %% [markdown]
# Setelah model clustering dilatih dan jumlah cluster optimal ditentukan, langkah selanjutnya adalah menampilkan hasil clustering melalui visualisasi.
# 
# Berikut adalah **rekomendasi** tahapannya.
# 1. Tampilkan hasil clustering dalam bentuk visualisasi, seperti grafik scatter plot atau 2D PCA projection.

# %%
# Menggunakan PCA untuk visualisasi data dalam 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Membuat DataFrame dengan hasil PCA
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

# Visualisasi hasil clustering dengan PCA
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=50)
plt.title('Visualisasi Cluster dengan PCA')
centroids_pca = pca.transform(kmeans_optimal.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## **e. Analisis dan Interpretasi Hasil Cluster**

# %% [markdown]
# ### Interpretasi Target

# %% [markdown]
# **Tutorial: Melakukan Inverse Transform pada Data Target Setelah Clustering**
# 
# Setelah melakukan clustering dengan model **KMeans**, kita perlu mengembalikan data yang telah diubah (normalisasi, standarisasi, atau label encoding) ke bentuk aslinya. Berikut adalah langkah-langkahnya.
# 
# ---
# 
# **1. Tambahkan Hasil Label Cluster ke DataFrame**
# Setelah mendapatkan hasil clustering, kita tambahkan label cluster ke dalam DataFrame yang telah dinormalisasi.
# 
# ```python
# df_normalized['Cluster'] = model_kmeans.labels_
# ```
# 
# Lakukan Inverse Transform pada feature yang sudah dilakukan Labelisasi dan Standararisasi. Berikut code untuk melakukannya:
# label_encoder.inverse_transform(X_Selected[['Fitur']])
# 
# Lalu masukkan ke dalam kolom dataset asli atau membuat dataframe baru
# ```python
# df_normalized['Fitur'] = label_encoder.inverse_transform(df_normalized[['Fitur']])
# ```
# Masukkan Data yang Sudah Di-Inverse ke dalam Dataset Asli atau Buat DataFrame Baru
# ```python
# df_original['Fitur'] = df_normalized['Fitur']
# ```

# %%
# Penambahan label cluster ke dalam DataFrame asli suda dilakukan pada bagian 6a sebelumnya.

# %% [markdown]
# ### Inverse Data Jika Melakukan Normalisasi/Standardisasi

# %% [markdown]
# Inverse Transform untuk Data yang Distandarisasi
# Jika data numerik telah dinormalisasi menggunakan StandardScaler atau MinMaxScaler, kita bisa mengembalikannya ke skala asli:
# ```python
# df_normalized[['Fitur_Numerik']] = scaler.inverse_transform(df_normalized[['Fitur_Numerik']])
# ```

# %%
# Type your code here

# %% [markdown]
# Setelah melakukan clustering, langkah selanjutnya adalah menganalisis karakteristik dari masing-masing cluster berdasarkan fitur yang tersedia.
# 
# Berikut adalah **rekomendasi** tahapannya.
# 1. Analisis karakteristik tiap cluster berdasarkan fitur yang tersedia (misalnya, distribusi nilai dalam cluster).
# 2. Berikan interpretasi: Apakah hasil clustering sesuai dengan ekspektasi dan logika bisnis? Apakah ada pola tertentu yang bisa dimanfaatkan?

# %%
# Analisis karakteristik tiap cluster
cluster_analysis_num = df.groupby('Cluster')[numerical_features].agg(['mean', 'min', 'max', 'median'])
print("\nAnalisis Karakteristik Numerik Tiap Cluster:")
print(cluster_analysis_num)

# Analisis distribusi fitur kategorikal di tiap cluster
for cluster in range(optimal_clusters):
    print(f"\nDistribusi Kategorikal untuk Cluster {cluster}:")
    for col in categorical_features:
        print(f"\n{col} di Cluster {cluster}:")
        print(df[df['Cluster'] == cluster][col].value_counts(normalize=True) * 100)

# Visualisasi karakteristik cluster untuk fitur numerik
for col in numerical_features:
    plt.figure(figsize=(12, 6))
    for cluster in range(optimal_clusters):
        sns.kdeplot(df[df['Cluster'] == cluster][col], label=f'Cluster {cluster}')
    plt.title(f'Distribusi {col} untuk Setiap Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualisasi distribusi kategorikal di tiap cluster
for col in categorical_features:
    plt.figure(figsize=(15, 10))
    for i, cluster in enumerate(range(optimal_clusters)):
        plt.subplot(optimal_clusters, 1, i + 1)
        cluster_data = df[df['Cluster'] == cluster]
        (cluster_data[col].value_counts(normalize=True) * 100).plot(kind='bar')
        plt.title(f'{col} di Cluster {cluster}')
        plt.ylabel('Persentase (%)')
    plt.tight_layout()
    plt.show()

# %%
# Visualisasi distribusi cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=df)
plt.title('Distribusi Cluster')
plt.show()

# %% [markdown]
# # Analisis Karakteristik Cluster dari Model K-Means
# 
# Berikut adalah analisis karakteristik untuk setiap cluster yang dihasilkan dari model KMeans pada dataset transaksi bank.
# 
# ## Cluster 0: Transaksi dengan Login Attempts Tinggi (3.78% dari total)
# - **Rata-rata TransactionAmount:** $276.32
# - **Rata-rata CustomerAge:** 44.54 tahun
# - **Rata-rata TransactionDuration:** 131.88 detik
# - **Rata-rata LoginAttempts:** 4.01
# - **Rata-rata AccountBalance:** $5,461.64
# - **Distribusi Pekerjaan:** Dokter (28.42%), Insinyur (26.32%), Mahasiswa (23.16%), Pensiunan (22.11%)
# - **Saluran Transaksi:** Online (38.95%), ATM (30.53%), Branch (30.53%)
# - **Analisis:** Cluster ini merupakan kelompok kecil dengan ciri khas jumlah percobaan login yang tinggi. Distribusi pekerjaan dan usia relatif seimbang, menunjukkan bahwa pola ini tidak terkait dengan demografi tertentu. Jumlah percobaan login yang tinggi bisa mengindikasikan nasabah yang kurang familiar dengan sistem perbankan digital atau potensi perilaku mencurigakan yang perlu dipantau oleh sistem keamanan bank.
# 
# ## Cluster 1: Transaksi Nasabah Pensiunan (28.03% dari total)
# - **Rata-rata TransactionAmount:** $210.64
# - **Rata-rata CustomerAge:** 62.77 tahun
# - **Rata-rata TransactionDuration:** 115.30 detik
# - **Rata-rata LoginAttempts:** 1.01
# - **Rata-rata AccountBalance:** $4,226.06
# - **Distribusi Pekerjaan:** Pensiunan (73.58%), Insinyur (17.33%), Dokter (9.09%)
# - **Saluran Transaksi:** ATM (35.37%), Branch (33.10%), Online (31.53%)
# - **Analisis:** Cluster ini didominasi oleh nasabah pensiunan dengan usia tinggi. Mereka cenderung melakukan transaksi bernilai menengah dengan sedikit kecenderungan menggunakan ATM. Pola ini mencerminkan preferensi generasi yang lebih tua terhadap layanan perbankan tradisional, meskipun nilai transaksi online juga cukup signifikan. Bank dapat menyesuaikan layanan ATM dan cabang untuk mengakomodasi kebutuhan segmen nasabah ini.
# 
# ## Cluster 2: Transaksi Mahasiswa dengan Saldo Rendah (28.94% dari total)
# - **Rata-rata TransactionAmount:** $238.29
# - **Rata-rata CustomerAge:** 25.66 tahun
# - **Rata-rata TransactionDuration:** 120.24 detik
# - **Rata-rata LoginAttempts:** 1.01
# - **Rata-rata AccountBalance:** $1,599.32
# - **Distribusi Pekerjaan:** Mahasiswa (78.27%), Insinyur (18.29%), Dokter (3.44%)
# - **Saluran Transaksi:** Branch (35.63%), ATM (33.29%), Online (31.09%)
# - **Analisis:** Cluster ini merepresentasikan segmen mahasiswa dengan usia muda dan saldo rekening rendah. Menariknya, meskipun mereka adalah generasi digital, preferensi mereka untuk transaksi cabang sedikit lebih tinggi. Ini mungkin menunjukkan kebutuhan konsultasi layanan perbankan karena keterbatasan pengalaman finansial. Bank dapat mengembangkan produk khusus untuk segmen ini dengan fokus pada edukasi finansial dan layanan dengan biaya rendah.
# 
# ## Cluster 3: Transaksi Dokter dengan Saldo Tinggi (29.54% dari total)
# - **Rata-rata TransactionAmount:** $224.84
# - **Rata-rata CustomerAge:** 45.97 tahun
# - **Rata-rata TransactionDuration:** 122.13 detik
# - **Rata-rata LoginAttempts:** 1.01
# - **Rata-rata AccountBalance:** $9,469.98
# - **Distribusi Pekerjaan:** Dokter (61.73%), Insinyur (36.79%), Mahasiswa (1.48%)
# - **Saluran Transaksi:** Branch (36.79%), Online (31.67%), ATM (31.54%)
# - **Analisis:** Cluster ini mewakili segmen profesional mapan, terutama dokter dan insinyur, dengan saldo rekening sangat tinggi. Meskipun memiliki kekayaan tinggi, nilai transaksi mereka relatif moderat, menunjukkan pola pengelolaan keuangan yang konservatif. Preferensi mereka untuk layanan cabang mungkin mencerminkan kebutuhan akan layanan perbankan yang lebih personal atau kompleks. Bank dapat mengoptimalkan layanan wealth management dan investasi untuk segmen nasabah premium ini.
# 
# ## Cluster 4: Transaksi Bernilai Tinggi (9.71% dari total)
# - **Rata-rata TransactionAmount:** $954.69
# - **Rata-rata CustomerAge:** 45.22 tahun
# - **Rata-rata TransactionDuration:** 118.08 detik
# - **Rata-rata LoginAttempts:** 1.02
# - **Rata-rata AccountBalance:** $4,769.25
# - **Distribusi Pekerjaan:** Insinyur (29.51%), Dokter (23.36%), Pensiunan (24.59%), Mahasiswa (22.54%)
# - **Saluran Transaksi:** Online (37.30%), ATM (32.38%), Branch (30.33%)
# - **Analisis:** Cluster ini menonjol dengan nilai transaksi yang sangat tinggi (lebih dari 4x cluster lain) namun tanpa kecenderungan demografis yang jelas. Distribusi pekerjaan dan saluran transaksi relatif merata, menunjukkan bahwa transaksi bernilai tinggi dilakukan oleh berbagai segmen nasabah. Preferensi yang sedikit lebih tinggi untuk transaksi online menunjukkan kenyamanan dalam melakukan transaksi besar secara digital. Bank dapat mempertimbangkan peningkatan limit transaksi online dan fitur keamanan tambahan untuk mengakomodasi perilaku ini.

# %% [markdown]
# # **7. Mengeksport Data**
# 
# Simpan hasilnya ke dalam file CSV.

# %%
# Menyimpan hasil clustering ke dalam file CSV
df.to_csv('bank_transactions_clustered.csv', index=False)
print("\nHasil clustering telah disimpan dalam file 'bank_transactions_clustered.csv'")