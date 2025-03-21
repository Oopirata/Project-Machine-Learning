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
#Type your code here

# %% [markdown]
# # **3. Memuat Dataset**

# %% [markdown]
# Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
# 
# Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.

# %%
#Type your code here

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
#Type your code here

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
#Type your code here

# %% [markdown]
# # **6. Pembangunan Model Clustering**

# %% [markdown]
# ## **a. Pembangunan Model Clustering**

# %% [markdown]
# Pada tahap ini, Anda membangun model clustering dengan memilih algoritma yang sesuai untuk mengelompokkan data berdasarkan kesamaan. Berikut adalah **rekomendasi** tahapannya.
# 1. Pilih algoritma clustering yang sesuai.
# 2. Latih model dengan data menggunakan algoritma tersebut.

# %%
#Type your code here

# %% [markdown]
# ## **b. Evaluasi Model Clustering**

# %% [markdown]
# Untuk menentukan jumlah cluster yang optimal dalam model clustering, Anda dapat menggunakan metode Elbow atau Silhouette Score.
# 
# Metode ini membantu kita menemukan jumlah cluster yang memberikan pemisahan terbaik antar kelompok data, sehingga model yang dibangun dapat lebih efektif. Berikut adalah **rekomendasi** tahapannya.
# 1. Gunakan Silhouette Score dan Elbow Method untuk menentukan jumlah cluster optimal.
# 2. Hitung Silhouette Score sebagai ukuran kualitas cluster.

# %%
#Type your code here

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
#Type your code here

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
# Type your code here


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
# Type your code here

# %% [markdown]
# Tulis hasil interpretasinya di sini.
# 1. Cluster 1:
# 2. Cluster 2:
# 3. Cluster 3:

# %% [markdown]
# # Contoh interpretasi [TEMPLATE]
# # Analisis Karakteristik Cluster dari Model KMeans
# 
# Berikut adalah analisis karakteristik untuk setiap cluster yang dihasilkan dari model KMeans.
# 
# ## Cluster 1:
# - **Rata-rata Annual Income (k$):** 48,260  
# - **Rata-rata Spending Score (1-100):** 56.48  
# - **Analisis:** Cluster ini mencakup pelanggan dengan pendapatan tahunan menengah dan tingkat pengeluaran yang cukup tinggi. Pelanggan dalam cluster ini cenderung memiliki daya beli yang moderat dan mereka lebih cenderung untuk membelanjakan sebagian besar pendapatan mereka.
# 
# ## Cluster 2:
# - **Rata-rata Annual Income (k$):** 86,540  
# - **Rata-rata Spending Score (1-100):** 82.13  
# - **Analisis:** Cluster ini menunjukkan pelanggan dengan pendapatan tahunan tinggi dan pengeluaran yang sangat tinggi. Pelanggan di cluster ini merupakan kelompok premium dengan daya beli yang kuat dan cenderung mengeluarkan uang dalam jumlah besar untuk produk atau layanan.
# 
# ## Cluster 3:
# - **Rata-rata Annual Income (k$):** 87,000  
# - **Rata-rata Spending Score (1-100):** 18.63  
# - **Analisis:** Cluster ini terdiri dari pelanggan dengan pendapatan tahunan yang tinggi tetapi pengeluaran yang rendah. Mereka mungkin memiliki kapasitas finansial yang baik namun tidak terlalu aktif dalam berbelanja. Ini bisa menunjukkan bahwa mereka lebih selektif dalam pengeluaran mereka atau mungkin lebih cenderung untuk menyimpan uang.

# %% [markdown]
# # **7. Mengeksport Data**
# 
# Simpan hasilnya ke dalam file CSV.

# %%



