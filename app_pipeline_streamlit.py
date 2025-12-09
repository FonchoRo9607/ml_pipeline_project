import streamlit as st
import pandas as pd
from app.data_cleaning.clean import clean_data
from app.training.train import train_model
from app.validation.validate import evaluate_model
from app.reporting.report import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    generate_pdf_report
)
import os

def main_pipeline(df, target_col):
    # Limpieza
    df = clean_data(df)

    # Definir target
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenamiento
    model = train_model(X_train, y_train)

    # Evaluación
    metrics = evaluate_model(model, X_test, y_test)

    # Predicciones
    y_pred = model.predict(X_test)

    # Gráficos
    os.makedirs("artifacts", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")

    y_pred_proba = None
    if hasattr(model, "predict_proba") and len(model.classes_) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, "artifacts/roc_curve.png")
        plot_precision_recall(y_test, y_pred_proba, "artifacts/precision_recall.png")

    # PDF
    generate_pdf_report(metrics, "artifacts/report.pdf")

    return metrics


# --- Interfaz Streamlit ---
st.title("📊 Pipeline ML con Reporte PDF")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Columnas detectadas:", df.columns.tolist())

    target_col = st.selectbox("Selecciona la columna objetivo", df.columns)

    if st.button("Ejecutar Pipeline"):
        metrics = main_pipeline(df, target_col)
        st.success("✅ Pipeline ejecutado correctamente. Se generó el PDF en la carpeta artifacts.")

        # Mostrar métricas en la app
        st.write("### Métricas de Validación")
        st.dataframe(pd.DataFrame(metrics).transpose().round(2))

        # Mostrar gráficos directamente en la app
        st.image("artifacts/confusion_matrix.png")
        if os.path.exists("artifacts/roc_curve.png"):
            st.image("artifacts/roc_curve.png")
        if os.path.exists("artifacts/precision_recall.png"):
            st.image("artifacts/precision_recall.png")