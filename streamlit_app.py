import streamlit as st
import cv2
from fastai.vision.all import *
from PIL import Image
import numpy as np
import time

# Google Drive 파일 ID
file_id = '1NKIhMhUeRC0vPptHwT4it-LMYhamVDyi'

# Google Drive에서 파일 다운로드 함수 (gdown 사용 가능)
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)
    
    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

# 모델의 분류 라벨 출력
labels = learner.dls.vocab
st.title(f"이미지 분류기 (Fastai) - 분류 라벨: {', '.join(labels)}")

# 카메라 스트림 시작
st.write("카메라를 통한 실시간 분류입니다.")

# OpenCV로 카메라 접근
cap = cv2.VideoCapture(0)  # 기본 카메라 장치 사용 (0번 카메라)

# 카메라에서 프레임을 주기적으로 읽어오기
while cap.isOpened():
    ret, frame = cap.read()  # 카메라에서 프레임 읽기
    if not ret:
        st.error("카메라에 접근할 수 없습니다.")
        break
    
    # 프레임을 PIL 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 포맷, Fastai는 RGB 사용
    img = PILImage.create(frame_rgb)  # Fastai가 처리할 수 있는 이미지 형식으로 변환

    # 예측 수행
    prediction, _, probs = learner.predict(img)

    # 카메라 화면 출력
    st.image(frame_rgb, caption=f"실시간 카메라 - 예측: {prediction}", use_column_width=True)

    # 확률 출력
    st.write("클래스별 확률:")
    for label, prob in zip(labels, probs):
        st.write(f"{label}: {prob:.4f}")

    # 프레임 사이의 지연 설정 (1초마다 업데이트)
    time.sleep(1)  # 1초마다 프레임 업데이트

# 카메라 자원 해제
cap.release()
cv2.destroyAllWindows()
