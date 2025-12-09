import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y preparación del dataset:
    - Elimina nulos y duplicados
    - Convierte la columna objetivo 'class' a binario (0/1)
    - Aplica One-Hot Encoding a las variables categóricas
    """

    # 1. Eliminar nulos y duplicados
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # 2. Convertir la columna objetivo a binario
    if "class" in df.columns:
        df["class"] = df["class"].map({
            "recurrence-events": 1,
            "false-recurrence-events": 0
        })

    # 3. Separar target antes de codificar
    target = df["class"]
    features = df.drop("class", axis=1)

    # 4. Codificar variables categóricas (One-Hot Encoding)
    features = pd.get_dummies(features, drop_first=True)

    # 5. Reconstruir DataFrame con target intacto
    df = features.copy()
    df["class"] = target

    return df

    
