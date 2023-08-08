import pymysql
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import joblib

# 1. 데이터 가져오기
conn = pymysql.connect(host='192.168.0.68', user='gulim', password='1234', db='gulim', charset='utf8')
cursor = conn.cursor(pymysql.cursors.DictCursor)

sql = "SELECT book_num, genre FROM book"
cursor.execute(sql)
books = cursor.fetchall()

conn.close()

# 데이터 전처리
genres = [row['genre'].split(',') for row in books]

# 빈 문자열 제거
genres = [[genre.strip() for genre in sublist if genre.strip()] for sublist in genres]

# 중복없는 장르 리스트 만들기
flattened_genres = [genre for sublist in genres for genre in sublist if genre]
unique_genres = list(set(flattened_genres))

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(unique_genres).reshape(-1, 1))  # 모든 유니크 장르에 대해 한 번만 fit

# 빈 장르 리스트를 가진 책들은 무시
genres_encoded = [encoder.transform(np.array(book_genres).reshape(-1, 1)).sum(axis=0)
                 for book_genres in genres if book_genres]

# k-NN 훈련
knn = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='cosine')
knn.fit(genres_encoded)

# 모델 저장
joblib.dump(knn, 'knn_model.pkl')

