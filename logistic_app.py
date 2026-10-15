import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
ADS_PATH = os.path.join(BASE_DIR, "advertising.csv")

LOG_FEATURES = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]
LOG_TARGET = "Clicked on Ad"

st.set_page_config(page_title="Logistic Regression — Ad Clicks", page_icon="📈", layout="wide")


@st.cache_data
def load_ads():
    return pd.read_csv(ADS_PATH)


@st.cache_resource
def train_logistic(df: pd.DataFrame):
    X = df[LOG_FEATURES]
    y = df[LOG_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return model, scaler, acc, cm


ads_df = load_ads()
model, scaler, acc, cm = train_logistic(ads_df)

st.title("📈 Will a user click on the ad?")
st.caption("Logistic regression trained on `advertising.csv`")

tab_predict, tab_explore, tab_data = st.tabs(["🔮 Predict", "📊 Explore", "🗂️ Data"])

with tab_predict:
    col_form, col_metrics = st.columns([2, 1])

    with col_form:
        st.subheader("Enter user / session details")
        c1, c2 = st.columns(2)
        with c1:
            time_on_site = st.number_input(
                "Daily Time Spent on Site (minutes)",
                min_value=0.0,
                value=float(ads_df["Daily Time Spent on Site"].mean()),
                step=0.5,
            )
            age = st.number_input(
                "Age", min_value=0, value=int(ads_df["Age"].mean()), step=1
            )
            income = st.number_input(
                "Area Income ($)",
                min_value=0.0,
                value=float(ads_df["Area Income"].mean()),
                step=500.0,
            )
        with c2:
            internet_usage = st.number_input(
                "Daily Internet Usage (minutes)",
                min_value=0.0,
                value=float(ads_df["Daily Internet Usage"].mean()),
                step=1.0,
            )
            male = st.selectbox("Gender", ["Female", "Male"])
            male_val = 1 if male == "Male" else 0

        if st.button("Predict Click", type="primary"):
            input_df = pd.DataFrame(
                [[time_on_site, age, income, internet_usage, male_val]],
                columns=LOG_FEATURES,
            )
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.success(f"✅ Likely to click the ad (probability: {proba:.1%})")
            else:
                st.info(f"❌ Unlikely to click the ad (probability: {proba:.1%})")

    with col_metrics:
        st.subheader("Model performance")
        st.metric("Accuracy", f"{acc:.1%}")
        st.caption(f"Trained on {len(ads_df):,} records")

        st.subheader("Confusion matrix")
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Click", "Click"],
            y=["No Click", "Click"],
            color_continuous_scale="Blues",
        )
        st.plotly_chart(cm_fig, width="stretch")

with tab_explore:
    st.subheader("Feature vs Clicked on Ad")
    feature_choice = st.selectbox("Choose a feature", LOG_FEATURES, key="ads_feature")
    fig = px.box(
        ads_df, x=LOG_TARGET, y=feature_choice, color=LOG_TARGET,
        title=f"{feature_choice} by {LOG_TARGET}",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Target distribution")
    st.plotly_chart(px.histogram(ads_df, x=LOG_TARGET), width="stretch")

with tab_data:
    st.subheader("Raw dataset")
    st.dataframe(ads_df, width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(ads_df.describe(), width="stretch")
