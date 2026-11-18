
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AUTO VALUE | AI 중고차 가격 분석",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CSS disabled for testing


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "used_car_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "used_cars.csv")



# STEP 02 - 차량 검색 및 가격 예측

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"모델 또는 데이터 로딩 실패: {e}")
    st.stop()

required = ["brand", "model", "model_year", "milage",
            "fuel_type", "transmission", "accident"]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"데이터에 필요한 컬럼이 없습니다: {missing}")
    st.stop()

st.markdown("""
<div class="hero">
    <div class="small">INTELLIGENT USED CAR VALUATION</div>
    <h1>AUTO VALUE</h1>
    <p>차량을 찾고, 데이터로 가치를 확인하세요.</p>
    <p>직접 학습한 머신러닝 모델을 활용한 중고차 가격 예측 서비스</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("01 · 내 차량 조건 검색")

    brands = sorted(df["brand"].dropna().astype(str).unique())
    brand = st.selectbox("제조사", brands)

    brand_df = df[df["brand"].astype(str) == brand]
    models = sorted(brand_df["model"].dropna().astype(str).unique())
    car_model = st.selectbox("차량 모델", models)

    model_df = brand_df[brand_df["model"].astype(str) == car_model]

    years = pd.to_numeric(
        model_df["model_year"], errors="coerce"
    ).dropna()

    default_year = int(years.median()) if len(years) else 2020
    year = st.number_input(
        "연식",
        min_value=1990,
        max_value=2025,
        value=max(1990, min(2025, default_year))
    )

    mileage_km = st.number_input(
        "주행거리 (km)",
        min_value=0,
        max_value=800000,
        value=48000,
        step=1000
    )

    fuel_options = sorted(
        df["fuel_type"].dropna().astype(str).unique()
    )
    fuel = st.selectbox("연료 종류", fuel_options)

    trans_options = sorted(
        df["transmission"].dropna().astype(str).unique()
    )
    transmission = st.selectbox("변속기", trans_options)

    accident_options = sorted(
        df["accident"].dropna().astype(str).unique()
    )
    accident = st.selectbox("사고 이력", accident_options)

    exchange_rate = st.number_input(
        "환율 설정 (1 USD → KRW)",
        min_value=1000,
        max_value=2000,
        value=1400,
        step=10
    )

    predict_button = st.button(
        "AI 예상 가격 분석하기",
        use_container_width=True
    )

with right:
    st.subheader("02 · 선택한 차량 정보")
    st.markdown(
        f"""
        <div class="panel">
            <div class="small">SELECTED VEHICLE</div>
            <h2>{brand} {car_model}</h2>
            <p>{year}년식 · {mileage_km:,} km</p>
            <p>연료: {fuel}</p>
            <p>변속기: {transmission}</p>
            <p>사고 이력: {accident}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "차량 사진은 추후 등록할 수 있습니다. "
        "사진 없이도 가격 예측 기능은 정상 작동합니다."
    )

if predict_button:
    input_data = pd.DataFrame([{
        "brand": brand,
        "model_year": int(year),
        "milage": float(mileage_km) / 1.609344,
        "fuel_type": fuel,
        "transmission": transmission,
        "accident": accident
    }])

    try:
        predicted_usd = float(model.predict(input_data)[0])
        predicted_krw = predicted_usd * exchange_rate

        st.divider()
        st.subheader("AI 중고차 가격 예측 결과")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("예상 가격 (USD)", f"${predicted_usd:,.0f}")
        with col2:
            st.metric("원화 환산 (KRW)", f"₩{predicted_krw:,.0f}")

        st.caption(
            "미국 중고차 데이터 기반 추정치입니다. "
            "원화 가격은 설정한 환율로 환산한 값이며, "
            "국내 실제 중고차 시세를 의미하지 않습니다."
        )

    except Exception as e:
        st.error(f"예측 실행 중 오류: {e}")



# STEP 03 - AI 분석 대시보드

st.divider()
st.subheader("03 · AI 분석 대시보드")

tab1, tab2, tab3 = st.tabs([
    "주행거리별 가격 시뮬레이션",
    "모델 성능 비교",
    "프로젝트 분석"
])

with tab1:
    st.markdown("#### 주행거리 변화에 따른 예상 가격")

    distances_km = [0, 15000, 30000, 45000, 60000, 90000, 120000]

    simulation_rows = []

    for km in distances_km:
        sample = pd.DataFrame([{
            "brand": brand,
            "model_year": int(year),
            "milage": float(km) / 1.609344,
            "fuel_type": fuel,
            "transmission": transmission,
            "accident": accident
        }])

        try:
            price_usd = float(model.predict(sample)[0])
            simulation_rows.append({
                "주행거리 (km)": km,
                "예상 가격 (USD)": price_usd
            })
        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
            break

    if simulation_rows:
        simulation_df = pd.DataFrame(simulation_rows)
        st.line_chart(
            simulation_df.set_index("주행거리 (km)")
        )

        st.caption(
            "차량 브랜드, 연식, 연료, 변속기, 사고 이력을 고정하고 "
            "주행거리만 변경한 머신러닝 예측 시뮬레이션입니다. "
            "실제 감가상각률을 의미하지 않습니다."
        )

with tab2:
    st.markdown("#### 머신러닝 모델 성능 비교")

    performance = pd.DataFrame({
        "모델": [
            "Random Forest",
            "XGBoost",
            "Tuned Random Forest"
        ],
        "MAE (USD)": [
            13859.36,
            14551.03,
            13875.35
        ],
        "RMSE (USD)": [
            30946.91,
            31007.05,
            29654.67
        ],
        "R²": [
            0.5938,
            0.5922,
            0.6270
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        performance.set_index("모델")[["RMSE (USD)"]]
    )

    st.info(
        "위 표는 모델 개발 과정의 검증 성능입니다. "
        "최종 테스트 데이터의 성능과는 다릅니다."
    )

    st.markdown("#### 최종 테스트 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", "$23,128")
    c2.metric("Test RMSE", "$153,245")
    c3.metric("Test R²", "0.097")

    st.caption(
        "고가 차량의 극단적인 가격 분포로 인해 "
        "최종 테스트 성능이 검증 성능보다 낮게 나타났습니다."
    )

with tab3:
    st.markdown("#### 프로젝트 개요")

    st.write(
        "중고차 데이터의 차량 브랜드, 연식, 주행거리, "
        "연료 종류, 변속기, 사고 이력 등을 이용하여 "
        "차량 가격을 예측하는 머신러닝 프로젝트입니다."
    )

    st.markdown("#### 사용 기술")
    st.write(
        "Python · Pandas · Scikit-learn · "
        "Random Forest · XGBoost · Streamlit"
    )

    st.markdown("#### 주요 분석 결과")
    st.write(
        "주행거리, 브랜드, 차량 연식 등이 가격 예측에 "
        "중요한 영향을 주는 변수로 분석되었습니다."
    )

    st.markdown("#### 한계점 및 개선 방향")
    st.write(
        "고가 차량과 이상치에 대한 예측 오차가 크며, "
        "미국 시장 기반 데이터이므로 국내 실거래 가격을 "
        "직접 반영하지 않습니다. 향후 더 많은 데이터와 "
        "가격 구간별 모델을 활용하여 성능을 개선할 수 있습니다."
    )

st.divider()
st.caption(
    "AUTO VALUE | Machine Learning Portfolio · "
    "학습 데이터 기반 참고용 가격 예측 서비스"
)
