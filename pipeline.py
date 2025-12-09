from app.data_ingestion.ingest import load_csv
from app.data_cleaning.clean import clean_data
from app.training.train import train_model
from app.validation.validate import evaluate_model
from app.reporting.report import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    generate_pdf_report
)

def main():
    # 1. Ingesta de datos
    df = load_csv("data/raw/breast-cancer.csv")
    print("Columnas originales:", df.columns.tolist())

    # 2. Limpieza y codificación
    df = clean_data(df)
    print("Columnas después de limpieza:", df.columns.tolist())

    # 3. Definir columna objetivo
    target_col = "class"
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # 4. Split en train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Entrenamiento del modelo
    model = train_model(X_train, y_train)

    # 6. Evaluación
    metrics = evaluate_model(model, X_test, y_test)

    # 7. Generación de predicciones
    y_pred = model.predict(X_test)

    # 8. Gráficos
    # Matriz de confusión (siempre disponible)
    plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")

    # Curvas ROC y Precisión-Recall (solo si el modelo es binario y tiene predict_proba)
    y_pred_proba = None
    if hasattr(model, "predict_proba") and len(model.classes_) == 2:
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            print("Error al generar probabilidades:", e)

    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, "artifacts/roc_curve.png")
        plot_precision_recall(y_test, y_pred_proba, "artifacts/precision_recall.png")

    # 9. Reporte PDF
    generate_pdf_report(metrics, "artifacts/report.pdf")


if __name__ == "__main__":
    main()