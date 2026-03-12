import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title("Auto Machine Learning CSV")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    # bersihkan data
    df = df.dropna()
    df = df.drop_duplicates()

    st.write("Dataset")
    st.dataframe(df)

    target = st.selectbox("Pilih Target", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    numeric_columns = X.select_dtypes(include=["int64","float64"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessing),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    st.success("Model berhasil dilatih")

    st.subheader("Input Data Baru")

    input_data = {}

    for col in X.columns:

        if col in numeric_columns:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    if st.button("Prediksi"):

        data_baru = pd.DataFrame([input_data])

        pred = model.predict(data_baru)[0]

        st.success(f"Hasil Prediksi: {pred}")
