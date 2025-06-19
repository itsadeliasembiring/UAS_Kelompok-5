# 1. Import Library dan Input Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load Dataset
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Books-Ratings.csv')
users = pd.read_csv('Users.csv')

# Menampilkan dataset
print("="*60)
print("1. DATASET")
print("="*60)

print("Dataset Books:")
display(books.head())
print(f"\nInfo Dataset Books: {books.shape[0]} rows, {books.shape[1]} columns")
print(books.info())

print("\nDataset Ratings:")
display(ratings.head())
print(f"\nInfo Dataset Ratings: {ratings.shape[0]} rows, {ratings.shape[1]} columns")
print(ratings.info())

print("\nDataset Users:")
display(users.head())
print(f"\nInfo Dataset Users: {users.shape[0]} rows, {users.shape[1]} columns")
print(users.info())

# 2. PreProcessing Data
print("\n" + "="*60)
print("2. PREPROCESSING DATA")
print("="*60)

# Gabung dataset terlebih dahulu
df = pd.merge(books, ratings, on='ISBN')
df = pd.merge(df, users, on='User-ID')
print(f"\nData setelah digabung: {df.shape[0]} rows, {df.shape[1]} columns")

# 2.1 CEK MISSING VALUE DAN HAPUS MISSING VALUE
print("\n" + "-"*50)
print("2.1 CEK MISSING VALUE DAN HAPUS MISSING VALUE")
print("-"*50)

print("\nMissing values sebelum preprocessing:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage (%)': missing_percentage
})
print(missing_df[missing_df['Missing Count'] > 0])

# Handle Age dengan mean imputation
print(f"\nMengisi missing value Age dengan rata-rata: {df['Age'].mean():.2f}")
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Hapus baris dengan missing value pada kolom penting lainnya
print("Menghapus baris dengan missing value pada kolom lainnya...")
df_before = df.shape[0]
df = df.dropna().reset_index(drop=True)
df_after = df.shape[0]
print(f"Data berkurang dari {df_before} menjadi {df_after} rows ({df_before - df_after} rows dihapus)")

print("\nMissing values setelah preprocessing:")
print(df.isnull().sum().sum(), "missing values tersisa")

# 2.2 CEK OUTLIER DAN HAPUS OUTLIER
print("\n" + "-"*50)
print("2.2 CEK OUTLIER DAN HAPUS OUTLIER")
print("-"*50)

# Cek outlier untuk Age
print("\nAnalisis outlier untuk kolom Age:")
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

outliers_age = df[(df['Age'] < lower_bound_age) | (df['Age'] > upper_bound_age)]
print(f"Q1: {Q1_age}, Q3: {Q3_age}, IQR: {IQR_age}")
print(f"Batas bawah: {lower_bound_age:.2f}, Batas atas: {upper_bound_age:.2f}")
print(f"Jumlah outlier Age: {len(outliers_age)} ({len(outliers_age)/len(df)*100:.2f}%)")

# Cek outlier untuk Book-Rating
print("\nAnalisis outlier untuk kolom Book-Rating:")
rating_stats = df['Book-Rating'].describe()
print(rating_stats)
print(f"Range valid rating: {df['Book-Rating'].min()} - {df['Book-Rating'].max()}")

# Hapus outlier Age (jika diperlukan - dalam kasus ini kita batasi umur 5-100 tahun)
print(f"\nMenghapus outlier Age (< 5 atau > 100 tahun)...")
df_before_outlier = df.shape[0]
df = df[(df['Age'] >= 5) & (df['Age'] <= 100)]
df_after_outlier = df.shape[0]
print(f"Data berkurang dari {df_before_outlier} menjadi {df_after_outlier} rows ({df_before_outlier - df_after_outlier} rows dihapus)")

# Hapus rating yang tidak valid (rating harus 0-10)
print(f"\nMenghapus rating tidak valid (di luar range 0-10)...")
df_before_rating = df.shape[0]
df = df[(df['Book-Rating'] >= 0) & (df['Book-Rating'] <= 10)]
df_after_rating = df.shape[0]
print(f"Data berkurang dari {df_before_rating} menjadi {df_after_rating} rows ({df_before_rating - df_after_rating} rows dihapus)")

# 2.3 NORMALISASI
print("\n" + "-"*50)
print("2.3 NORMALISASI")
print("-"*50)

# Encode User-ID dan ISBN
print("Melakukan encoding untuk User-ID dan ISBN...")
user_ids = df['User-ID'].unique()
isbn_list = df['ISBN'].unique()
user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
isbn_to_idx = {isbn: idx for idx, isbn in enumerate(isbn_list)}
df['user_idx'] = df['User-ID'].map(user_to_idx)
df['isbn_idx'] = df['ISBN'].map(isbn_to_idx)
print(f"Total unique users: {len(user_ids)}")
print(f"Total unique books: {len(isbn_list)}")

# Normalisasi rating menggunakan MinMaxScaler
print("\nNormalisasi Book-Rating menggunakan MinMaxScaler...")
print(f"Rating sebelum normalisasi - Min: {df['Book-Rating'].min()}, Max: {df['Book-Rating'].max()}")
scaler = MinMaxScaler()
df['Book-Rating'] = scaler.fit_transform(df[['Book-Rating']])
print(f"Rating setelah normalisasi - Min: {df['Book-Rating'].min():.3f}, Max: {df['Book-Rating'].max():.3f}")

# Normalisasi Age (opsional)
print("\nNormalisasi Age menggunakan MinMaxScaler...")
print(f"Age sebelum normalisasi - Min: {df['Age'].min():.2f}, Max: {df['Age'].max():.2f}")
age_scaler = MinMaxScaler()
df['Age_normalized'] = age_scaler.fit_transform(df[['Age']])
print(f"Age setelah normalisasi - Min: {df['Age_normalized'].min():.3f}, Max: {df['Age_normalized'].max():.3f}")

print("\n" + "-"*50)
print("RINGKASAN PREPROCESSING")
print("-"*50)
print(f"Dataset final: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Unique users: {df['User-ID'].nunique()}")
print(f"Unique books: {df['ISBN'].nunique()}")
print(f"Rating range: {df['Book-Rating'].min():.3f} - {df['Book-Rating'].max():.3f}")
print(f"Age range: {df['Age'].min():.2f} - {df['Age'].max():.2f} tahun")
print("Missing values:", df.isnull().sum().sum())

print("\nSample data setelah preprocessing:")
display(df[['User-ID', 'ISBN', 'Book-Title', 'Book-Author', 'Book-Rating', 'Age', 'user_idx', 'isbn_idx']].head())

# 3. Similarity Calculation
print("\n" + "="*60)
print("3. SIMILARITY CALCULATION")
print("="*60)

# Buat matriks rating sparse
sparse_matrix = csr_matrix((df['Book-Rating'], (df['user_idx'], df['isbn_idx'])))
print(f"Ukuran matriks sparse: {sparse_matrix.shape} (users x books)")
print(f"Sparsity: {(1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100:.2f}%")

# Ambil hanya buku dengan minimal 10 rating untuk mengurangi noise
print("\nMemfilter buku dengan minimal 10 rating...")
popular_books = df.groupby('ISBN').filter(lambda x: len(x) >= 10)
popular_isbns = popular_books['ISBN'].unique()
popular_isbn_idx = [isbn_to_idx[isbn] for isbn in popular_isbns]
sparse_popular = sparse_matrix[:, popular_isbn_idx]
print(f"Buku populer: {len(popular_isbns)} dari {len(isbn_list)} total buku")
print(f"Ukuran matriks setelah filtering: {sparse_popular.shape}")

# a. Cosine Similarity
print("\nMenghitung Cosine Similarity...")
cos_sim = cosine_similarity(sparse_popular.T)
cos_sim_df = pd.DataFrame(cos_sim, index=popular_isbns, columns=popular_isbns)
print(f"Cosine similarity matrix shape: {cos_sim_df.shape}")

# b. Euclidean Distance (dijadikan similarity: sim = 1 / (1 + distance))
print("\nMenghitung Euclidean Similarity...")
euc_dist = euclidean_distances(sparse_popular.T)
euc_sim = 1 / (1 + euc_dist)
euc_sim_df = pd.DataFrame(euc_sim, index=popular_isbns, columns=popular_isbns)
print(f"Euclidean similarity matrix shape: {euc_sim_df.shape}")

# c. Manhattan Distance (L1 norm) - dijadikan similarity: sim = 1 / (1 + distance)
print("\nMenghitung Manhattan Similarity...")
manhattan_dist = manhattan_distances(sparse_popular.T)
manhattan_sim = 1 / (1 + manhattan_dist)
manhattan_sim_df = pd.DataFrame(manhattan_sim, index=popular_isbns, columns=popular_isbns)
print(f"Manhattan similarity matrix shape: {manhattan_sim_df.shape}")

print("\nContoh Similarity Values (5x5 sample):")
print("\nCosine Similarity:")
print(cos_sim_df.iloc[:5, :5].round(4))
print("\nEuclidean Similarity:")
print(euc_sim_df.iloc[:5, :5].round(4))
print("\nManhattan Similarity:")
print(manhattan_sim_df.iloc[:5, :5].round(4))

# 4. Collaborative Filtering: Item-Based
def get_recommendations(isbn, similarity_df, n=5):
    """Mendapatkan rekomendasi buku berdasarkan similarity matrix"""
    if isbn not in similarity_df.index:
        return []
    sim_scores = similarity_df[isbn].sort_values(ascending=False)[1:n+1]
    return sim_scores.index.tolist()

# 5. Evaluasi: 20 User dan rekomendasi untuk ketiga metrik similarity
print("\n" + "="*60)
print("4. EVALUASI SISTEM REKOMENDASI")
print("="*60)

random.seed(42)  # Untuk reproducibility
random_users = random.sample(list(df['User-ID'].unique()), 20)
print(f"Mengevaluasi sistem untuk {len(random_users)} user random...")

# Mapping ISBN ke Book-Title
isbn_to_title = pd.Series(books['Book-Title'].values, index=books['ISBN']).to_dict()

# Dictionary untuk menyimpan hasil evaluasi setiap metrik
evaluation_results = {
    'Cosine Similarity': {'success_count': 0, 'similarity_df': cos_sim_df, 'total_recommendations': 0},
    'Euclidean Similarity': {'success_count': 0, 'similarity_df': euc_sim_df, 'total_recommendations': 0},
    'Manhattan Similarity': {'success_count': 0, 'similarity_df': manhattan_sim_df, 'total_recommendations': 0}
}

for i, user in enumerate(random_users, 1):
    user_data = df[df['User-ID'] == user]
    if len(user_data) == 0:
        continue

    user_books = user_data['ISBN'].tolist()
    
    print(f"\n{'='*80}")
    print(f"EVALUASI USER {i}/20 - User-ID: {user}")
    print(f"{'='*80}")
    print(f"Buku yang pernah dibaca: {len(user_books)} buku")
    
    # Tampilkan semua buku yang pernah dibaca
    for j, isbn in enumerate(user_books):
        title = isbn_to_title.get(isbn, 'Judul tidak ditemukan')
        print(f"  {j+1}. {title[:50]}..." if len(title) > 50 else f"  {j+1}. {title}")
    
    # Evaluasi untuk setiap metrik similarity
    for method_name, method_data in evaluation_results.items():
        similarity_df = method_data['similarity_df']
        recommended_books = []

        # Generate rekomendasi berdasarkan similarity matrix
        for book in user_books:
            if book in similarity_df.index:
                recs = get_recommendations(book, similarity_df, n=3)
                recommended_books.extend(recs)

        # Hapus duplikat dan buku yang sudah dibaca
        recommended_books = list(set(recommended_books))
        recommended_books = [book for book in recommended_books if book not in user_books]
        
        method_data['total_recommendations'] += len(recommended_books)
        
        # Untuk evaluasi sederhana, kita anggap sukses jika ada rekomendasi yang dihasilkan
        success = len(recommended_books) > 0
        if success:
            method_data['success_count'] += 1

        # Tampilkan hasil per metrik
        print(f"\n--- {method_name} ---")
        if recommended_books:
            print(f"Rekomendasi: {len(recommended_books)} buku")
            # Tampilkan SEMUA rekomendasi
            for j, isbn in enumerate(recommended_books):
                title = isbn_to_title.get(isbn, 'Judul tidak ditemukan')
                print(f"  {j+1}. {title[:50]}..." if len(title) > 50 else f"  {j+1}. {title}")
        else:
            print("Rekomendasi: Tidak ada (buku tidak ditemukan dalam similarity matrix)")
        print(f"Jumlah rekomendasi: {len(recommended_books)} buku")
        
# Tampilkan ringkasan evaluasi akhir
print(f"\n{'='*80}")
print("RINGKASAN EVALUASI AKHIR")
print(f"{'='*80}")

results_summary = []
for method_name, method_data in evaluation_results.items():
    success_count = method_data['success_count']
    total_recs = method_data['total_recommendations']
    success_rate = (success_count / 20) * 100
    avg_recs = total_recs / 20
    
    results_summary.append({
        'Method': method_name,
        'Success Rate': f"{success_count}/20 ({success_rate:.1f}%)",
        'Avg Recommendations': f"{avg_recs:.1f} buku/user"
    })
    
    print(f"{method_name:20} : {success_count}/20 user ({success_rate:.1f}%) berhasil mendapat rekomendasi")
    print(f"{'':20}   Rata-rata {avg_recs:.1f} rekomendasi per user")

# Menentukan metrik terbaik
best_method = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['success_count'])
best_score = evaluation_results[best_method]['success_count']
print(f"\n🏆 Metrik terbaik: {best_method}")
print(f"   Berhasil memberikan rekomendasi untuk {best_score}/20 user")

# Tampilkan tabel ringkasan
print(f"\n{'='*80}")
print("TABEL PERBANDINGAN METRIK")
print(f"{'='*80}")
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))