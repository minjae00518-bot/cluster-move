import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------------------------------------------------
# 1. 페이지 설정 및 메인 제목/아이콘 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.markdown("영화 데이터를 활용해 속성별 표준화 후 **K-Means 알고리즘**으로 영화를 3가지 유형(㉮, ㉯, ㉰)으로 군집화합니다.")

# 데이터 경로 (실제 데이터셋 URL 지정)
DATA_URL = "https://raw.githubusercontent.com/..." 

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
    except Exception:
        # 데이터 연동 전 예시 실행을 위한 구조 샘플링
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'movieCd': [f'2023{i:04d}' for i in range(n)],
            'movieNm': [f'영화_{i+1}' for i in range(n)],
            'openDt': ['2023-01-01'] * n,
            'first_scrn': np.random.randint(10, 2000, n),
            'total_audi': np.random.randint(1000, 10000000, n),
            'days_in_top10': np.random.randint(1, 60, n),
            'long_run': np.random.uniform(0.1, 5.0, n),
            'first_week_audi': np.random.randint(0, 2000000, n)
        })
    return df

raw_df = load_data(DATA_URL)

# ---------------------------------------------------------
# 2. 데이터 전처리
# ---------------------------------------------------------
# 4개의 핵심 분석 속성
feature_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'long_run']
feature_labels = {
    'first_scrn': '스크린 수 (first_scrn)',
    'total_audi': '누적 관객 (total_audi)',
    'days_in_top10': '10위권 일수 (days_in_top10)',
    'long_run': '롱런 지수 (long_run)'
}

# 네 속성 중 결측치가 있는 행 및 첫 주 관객이 0인 영화 제거
df = raw_df.dropna(subset=feature_cols).copy()
if 'first_week_audi' in df.columns:
    df = df[df['first_week_audi'] > 0]

# ---------------------------------------------------------
# 3. 속성 선택 및 K-Means 클러스터링
# ---------------------------------------------------------
st.write("---")
st.subheader("⚙️ 묶음에 사용할 속성 선택")

selected_features = st.multiselect(
    "군집화(K-Means)에 사용할 속성을 선택해 주세요:",
    options=feature_cols,
    default=feature_cols,
    format_func=lambda x: feature_labels.get(x, x)
)

if not selected_features:
    st.error("최소 한 개 이상의 속성을 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

# 고른 속성 표준화 (StandardScaler)
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df[selected_features])

# K-평균 알고리즘 (3개 묶음)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_num'] = kmeans.fit_predict(scaled_matrix)

# 묶음 번호 변환 (0, 1, 2 -> ㉮, ㉯, ㉰)
cluster_mapping = {0: '㉮', 1: '㉯', 2: '㉰'}
df['cluster'] = df['cluster_num'].map(cluster_mapping)

# ---------------------------------------------------------
# 4. 시각화 (2차원 및 3차원 산점도)
# ---------------------------------------------------------
st.write("---")
st.subheader("📊 시각화 분석")

# (1) 2차원 산점도
st.markdown("#### 📍 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_2d = st.selectbox("가로축(X축) 속성:", feature_cols, index=0, format_func=lambda x: feature_labels[x], key='2d_x')
with col2:
    y_2d = st.selectbox("세로축(Y축) 속성:", feature_cols, index=1, format_func=lambda x: feature_labels[x], key='2d_y')

fig_2d = px.scatter(
    df,
    x=x_2d,
    y=y_2d,
    color='cluster',
    hover_data=['movieNm', 'total_audi', 'first_scrn'],
    title=f"2D 산점도: {feature_labels[x_2d]} vs {feature_labels[y_2d]}",
    labels=feature_labels,
    color_discrete_map={'㉮': '#EF553B', '㉯': '#636EFA', '㉰': '#00CC96'}
)
st.plotly_chart(fig_2d, use_container_width=True)

# (2) 3차원 산점도
st.markdown("#### 🧊 3차원 산점도")
col3, col4, col5 = st.columns(3)
with col3:
    x_3d = st.selectbox("X축 속성:", feature_cols, index=0, format_func=lambda x: feature_labels[x], key='3d_x')
with col4:
    y_3d = st.selectbox("Y축 속성:", feature_cols, index=1, format_func=lambda x: feature_labels[x], key='3d_y')
with col5:
    z_3d = st.selectbox("Z축 속성:", feature_cols, index=2, format_func=lambda x: feature_labels[x], key='3d_z')

fig_3d = px.scatter_3d(
    df,
    x=x_3d,
    y=y_3d,
    z=z_3d,
    color='cluster',
    hover_data=['movieNm', 'total_audi'],
    title=f"3D 산점도: {feature_labels[x_3d]} × {feature_labels[y_3d]} × {feature_labels[z_3d]}",
    labels=feature_labels,
    color_discrete_map={'㉮': '#EF553B', '㉯': '#636EFA', '㉰': '#00CC96'}
)
st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# 5. 묶음별 통계 및 대표 영화
# ---------------------------------------------------------
st.write("---")
st.subheader("📋 묶음별 통계 요약 및 상위 영화")

# 묶음별 영화 편수 및 네 속성의 평균(원래 단위)
summary_table = df.groupby('cluster').agg(
    영화편수=('movieCd', 'count'),
    스크린수_평균=('first_scrn', 'mean'),
    누적관객_평균=('total_audi', 'mean'),
    TOP10일수_평균=('days_in_top10', 'mean'),
    롱런지수_평균=('long_run', 'mean')
).reset_index()

st.markdown("##### 1) 묶음별 영화 편수 및 4개 속성 평균값 (원래 단위)")
st.dataframe(
    summary_table.style.format({
        '영화편수': '{:,}편',
        '스크린수_평균': '{:,.1f}',
        '누적관객_평균': '{:,.1f}',
        'TOP10일수_평균': '{:,.1f}',
        '롱런지수_평균': '{:,.2f}'
    }),
    use_container_width=True
)

# 묶음별 누적 관객 수가 많은 순서로 상위 5편
st.markdown("##### 2) 묶음별 누적 관객 수 Top 5 영화")
cluster_cols = st.columns(3)

for idx, cluster_label in enumerate(['㉮', '㉯', '㉰']):
    with cluster_cols[idx]:
        st.markdown(f"### 📌 묶음 {cluster_label}")
        top5_movies = (
            df[df['cluster'] == cluster_label]
            .sort_values(by='total_audi', ascending=False)
            .head(5)[['movieNm', 'total_audi', 'first_scrn', 'days_in_top10']]
            .rename(columns={
                'movieNm': '영화명',
                'total_audi': '누적 관객',
                'first_scrn': '스크린 수',
                'days_in_top10': '10위권 일수'
            })
        )
        st.dataframe(
            top5_movies.style.format({'누적 관객': '{:,}', '스크린 수': '{:,}'}),
            use_container_width=True
        )
