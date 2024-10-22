import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
import plotly.graph_objects as go

# Google Drive 파일 ID
file_id = '1NKIhMhUeRC0vPptHwT4it-LMYhamVDyi'

# Google Drive에서 파일 다운로드 함수
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

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    # 업로드된 이미지 보여주기
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # Fastai에서 예측을 위해 이미지를 처리
    img = PILImage.create(uploaded_file)

    # 예측 수행
    prediction, _, probs = learner.predict(img)

    # 결과 출력
    st.write(f"예측된 클래스: {prediction}")

    # 확률 막대 그래프 생성
    fig = go.Figure([go.Bar(x=labels, y=probs, text=[f'{p:.4f}' for p in probs], 
                            textposition='auto', marker_color='lightblue')])

    # 레이아웃 설정
    fig.update_layout(
        title='분류 확률',
        xaxis_title='클래스',
        yaxis_title='확률',
        yaxis_range=[0, 1],  # 확률이 0에서 1 사이에 있으므로
        plot_bgcolor='rgba(0,0,0,0)'  # 배경 투명하게 설정
    )

    # 그래프 표시
    st.plotly_chart(fig)
