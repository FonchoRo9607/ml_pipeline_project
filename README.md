# ML Pipeline con Streamlit

Este proyecto implementa un pipeline de Machine Learning en Python que genera métricas, gráficos y un informe PDF. Además, incluye una interfaz web con Streamlit para que cualquier usuario pueda subir su propio archivo CSV y obtener resultados de manera interactiva.

# Ejecución en Consola

Coloca tu archivo CSV en la carpeta data/raw/.

Ejecuta el pipeline desde la terminal:

```
python pipeline.py --csv data/raw/breast-cancer.csv --target class

```
El informe PDF y los gráficos se generarán en la carpeta artifacts/.

#  Ejecución con Streamlit

Ejecuta la aplicación web:
```
streamlit run app_pipeline_streamlit.py

```
Se abrirá una interfaz en tu navegador (por defecto en http://localhost:8501).

Sube tu archivo CSV desde la interfaz.

Selecciona la columna objetivo.

Haz clic en Ejecutar Pipeline.

Verás las métricas y gráficos directamente en la web, y se generará un PDF en artifacts/.

# Estructura del Proyecto

```
ml_pipeline_project/
│
├── app_pipeline_streamlit.py   # Interfaz web con Streamlit
├── pipeline.py                 # Versión consola
│
├── app/
│   ├── data_ingestion/ingest.py
│   ├── data_cleaning/clean.py
│   ├── training/train.py
│   ├── validation/validate.py
│   └── reporting/report.py
│
├── data/raw/                   # Datasets de ejemplo
└── artifacts/                  # PDFs y gráficos generados
```

#  Dependencias

Instala las librerías necesarias:

```
pip install -r requirements.txt

```
Ejemplo de requirements.txt:

- pandas
- scikit-learn
- matplotlib
- seaborn
- reportlab
- streamlit

#  Bilingual Instructions

Run in Console

```

python pipeline.py --csv data/raw/breast-cancer.csv --target class

```
Generates PDF and plots in artifacts/.

Run with Streamlit
```
streamlit run app_pipeline_streamlit.py
```
Open browser at http://localhost:8501, upload CSV, select target column, run pipeline.

#  Features

Flexible: works with any CSV dataset.

Generates confusion matrix, ROC curve, Precision-Recall curve.

Produces professional PDF reports.

Interactive web interface with Streamlit.

#  Notas

Usa .gitignore para excluir artifacts/ y archivos temporales.

Incluye un dataset de ejemplo (breast-cancer.csv) en data/raw/.

El proyecto está listo para subir a GitHub y compartir.
