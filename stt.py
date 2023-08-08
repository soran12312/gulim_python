from flask import Flask, request
from google.cloud import speech_v1p1beta1 as speech
from flask_cors import CORS
import os
import subprocess
import pymysql
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\python\gulim_STT2\heroic-oven-393800-7394c2e869f6.json"

app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['GET'])
def recommend_books_for_user():
    # 파라미터에서 아이디 얻어옴
    user_id = request.args.get('user_id')

    # 모델 불러오기
    knn_loaded = joblib.load('knn_model.pkl')

    # 유저 데이터 가져오기
    conn = pymysql.connect(host='192.168.0.68', user='gulim', password='1234', db='gulim', charset='utf8')
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    sql = f"SELECT play_genre, want_genre FROM survey WHERE id = '{user_id}'"
    cursor.execute(sql)
    users = cursor.fetchall()

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

    # 데이터 전처리
    user_play_genres = [genre for row in users for genre in row['play_genre'].split('/') if row['play_genre'] != '없음']
    user_want_genres = [genre for row in users for genre in row['want_genre'].split('/') if row['want_genre'] != '없음']

    user_play_genres = list(set(user_play_genres))
    user_want_genres = list(set(user_want_genres))

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(unique_genres).reshape(-1, 1))

    user_play_encoded = encoder.transform(np.array(user_play_genres).reshape(-1, 1)).sum(axis=0)
    user_want_encoded = encoder.transform(np.array(user_want_genres).reshape(-1, 1)).sum(axis=0) * 1.5  # 가중치 부여

    combined_encoded = user_play_encoded + user_want_encoded
    combined_encoded = combined_encoded.reshape(1, -1)

    # 추천 제공
    distances, indices = knn_loaded.kneighbors(combined_encoded, n_neighbors=4)

    book_nums = [row['book_num'] for row in books]

    result = [book_nums[i] for i in indices[0]]

    print(result)

    return result


@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 받아온 오디오 데이터
    audio_data = request.files['audio'].read()

    # 오디오 데이터를 파일로 저장
    with open('received_audio.ogg', 'wb') as f:
        f.write(audio_data)

    # ffmpeg를 이용하여 오디오를 LINEAR16로 변환
    command = ['D:\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe', '-i', 'received_audio.ogg', '-f', 'wav',
               '-acodec', 'pcm_s16le', '-ar', '16000', 'received_audio.wav']
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open('received_audio.wav', 'rb') as f:
        audio_data = f.read()

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)

    print('Response: %s', response)

    # 첫 번째 결과만 반환
    for result in response.results:
        print(result.alternatives[0].transcript)
        return result.alternatives[0].transcript

    return "No transcript available"

if __name__ == "__main__":
    app.run(host='192.168.0.68',port=5000, debug=True, ssl_context=('key/.cert.pem', 'key/key.pem'))

# Use ffmpeg to convert the m4a file to wav
# command = ['D:\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe', '-i', './news.m4a', '-f', 'wav', '-acodec',
#            'pcm_s16le', '-ar', '16000', './news.wav']
# subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
# # Read the converted audio file
# with open('./news.wav', 'rb') as f:
#     audio_data = f.read()
#
# # Setup the Google Cloud Speech client
# client = speech.SpeechClient()
# audio = speech.RecognitionAudio(content=audio_data)
# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz=16000,
#     language_code="ko-KR",  # Set this to the language of the audio file
# )
#
# # Call the Google Cloud Speech API
# response = client.recognize(config=config, audio=audio)
#
# # Print the transcription of the audio file
# for result in response.results:
#     print(result.alternatives[0].transcript)