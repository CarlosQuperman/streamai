import streamlit as st
import pandas as pd
import fastbook
from fastbook import *

# 제목
st.title("CSV 파일 업로드 및 데이터 표시")

# 파일 업로드 컴포넌트
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")

# 파일이 업로드되었는지 확인
if uploaded_file is not None:
    # 판다스를 이용해 파일 읽기
    df = pd.read_csv(uploaded_file)
    
    # 데이터프레임 표시
    st.write("업로드된 CSV 파일의 데이터:")
    st.dataframe(df)
    
    # 기본 통계 정보 표시
    st.write("기본 통계 정보:")
    st.write(df.describe())
    
    # 데이터프레임의 정보 출력
    st.write("데이터프레임의 정보:")
    st.write(df.info())
