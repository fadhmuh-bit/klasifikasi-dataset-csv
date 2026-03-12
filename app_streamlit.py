import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("Auto Machine Learning CSV")

file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset")
    st.dataframe(df)

    # pilih target
    target = st.selectbox("Pilih Kolom Target", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # deteksi tipe kolom
    numeric_columns = X.select_dtypes(include=["int64","float64"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    # preprocessing
    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessing),
            ("model", LogisticRegression())
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    st.subheader("Input Data Baru")

    input_data = {}

    for col in X.columns:

        if col in numeric_columns:
            input_data[col] = st.slider(
                col,
                int(df[col].min()),
                int(df[col].max()),
                int(df[col].median())
            )

        else:
            input_data[col] = st.selectbox(
                col,
                df[col].unique()
            )

    if st.button("Prediksi"):

        data_baru = pd.DataFrame([input_data])

        prediksi = model.predict(data_baru)[0]

        st.success(f"Hasil Prediksi: {prediksi}")
        st.balloons()