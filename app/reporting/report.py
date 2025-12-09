from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os

# --- Gráficos ---
def plot_confusion_matrix(y_true, y_pred, output_path="artifacts/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.title("Matriz de Confusión")
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_path="artifacts/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall(y_true, y_pred_proba, output_path="artifacts/precision_recall.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precisión vs Recall")
    plt.savefig(output_path)
    plt.close()

# --- Reporte PDF ---
def generate_pdf_report(metrics: dict, output_path="artifacts/report.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    flow = []

    # Título
    flow.append(Paragraph("Informe del Pipeline de ML", styles["Title"]))
    flow.append(Spacer(1, 12))

    # Tabla de métricas
    df = pd.DataFrame(metrics).transpose().round(2)
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    flow.append(Paragraph("Métricas de Validación", styles["Heading2"]))
    flow.append(table)
    flow.append(Spacer(1, 20))

    # Imágenes (solo si existen)
    for title, path in [
        ("Matriz de Confusión", "artifacts/confusion_matrix.png"),
        ("Curva ROC", "artifacts/roc_curve.png"),
        ("Precisión vs Recall", "artifacts/precision_recall.png")
    ]:
        if os.path.exists(path):
            flow.append(Paragraph(title, styles["Heading2"]))
            flow.append(Image(path, width=400, height=200))
            flow.append(Spacer(1, 20))
        else:
            flow.append(Paragraph(f"{title} no disponible (no se generó)", styles["Normal"]))
            flow.append(Spacer(1, 12))

    # Generar PDF
    doc.build(flow)
