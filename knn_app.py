import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
TITANIC_PATH = os.path.join(BASE_DIR, "titanic_test.csv")

KNN_NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
KNN_FEATURES = KNN_NUMERIC + ["Sex_male", "Embarked_Q", "Embarked_S"]

st.set_page_config(page_title="KNN — Titanic Nearest Passengers", page_icon="🧭", layout="wide")


@st.cache_data
def load_titanic():
    df = pd.read_csv(TITANIC_PATH)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


@st.cache_resource
def build_knn(df: pd.DataFrame):
    encoded = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    # ensure expected dummy columns exist even if a category is absent
    for col in ["Sex_male", "Embarked_Q", "Embarked_S"]:
        if col not in encoded.columns:
            encoded[col] = 0

    X = encoded[KNN_FEATURES].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(X_scaled)
    return knn, scaler


titanic_df = load_titanic()
knn, knn_scaler = build_knn(titanic_df)

st.title("🧭 Find similar Titanic passengers (KNN)")
st.caption(
    "`titanic_test.csv` has no `Survived` label, so this uses unsupervised "
    "K-Nearest Neighbors to find the most similar passengers to a profile you enter."
)

tab_predict, tab_explore, tab_data = st.tabs(["🔮 Find Neighbors", "📊 Explore", "🗂️ Data"])

with tab_predict:
    st.subheader("Enter a passenger profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
    with c2:
        age = st.number_input("Age", min_value=0.0, value=float(titanic_df["Age"].median()), step=1.0)
        sibsp = st.number_input("Siblings/Spouses aboard (SibSp)", min_value=0, value=0, step=1)
    with c3:
        parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, value=0, step=1)
        fare = st.number_input("Fare", min_value=0.0, value=float(titanic_df["Fare"].median()), step=1.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    k = st.slider("Number of neighbors (K)", min_value=1, max_value=10, value=5)

    if st.button("Find Nearest Passengers", type="primary"):
        input_row = pd.DataFrame(
            [[pclass, age, sibsp, parch, fare,
              1 if sex == "male" else 0,
              1 if embarked == "Q" else 0,
              1 if embarked == "S" else 0]],
            columns=KNN_FEATURES,
        )
        input_scaled = knn_scaler.transform(input_row)
        distances, indices = knn.kneighbors(input_scaled, n_neighbors=k)

        st.success(f"Top {k} most similar passengers:")
        result_df = titanic_df.iloc[indices[0]].copy()
        result_df.insert(0, "Distance", np.round(distances[0], 3))
        st.dataframe(
            result_df[["Distance", "PassengerId", "Name", "Pclass", "Sex", "Age",
                       "SibSp", "Parch", "Fare", "Embarked"]],
            width="stretch",
        )

with tab_explore:
    st.subheader("Feature distribution")
    feature_choice = st.selectbox("Choose a feature", KNN_NUMERIC, key="titanic_feature")
    st.plotly_chart(px.histogram(titanic_df, x=feature_choice, color="Sex"), width="stretch")

    st.subheader("Age vs Fare by Pclass")
    st.plotly_chart(
        px.scatter(titanic_df, x="Age", y="Fare", color="Pclass", symbol="Sex",
                   hover_data=["Name"]),
        width="stretch",
    )

with tab_data:
    st.subheader("Raw dataset")
    st.dataframe(titanic_df, width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(titanic_df.describe(), width="stretch")
