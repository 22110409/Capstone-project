import pandas as pd

def clean_dataframe(df, feature_cols, fill_value=0):
    # Columns to drop if they exist
    cols_to_drop = [
        "Flow ID", "Src IP", "Dst IP",
        "Src Port", "Dst Port", "Timestamp",
        "Active Mean", "Bwd PSH Flags", "Bwd Byts/b Avg",
        "Active Std", "Active Max", "Fwd URG Flags",
        "Bwd Pkt Len Min", "Idle Max", "Bwd Pkts/b Avg",
        "Idle Mean", "Fwd Pkts/b Avg", "Idle Std",
        "Fwd Byts/b Avg", "Active Min", "Fwd Blk Rate Avg",
        "Bwd URG Flags", "Idle Min", "Pkt Len Min",
        "CWE Flag Count", "Fwd Pkt Len Min",
        "Bwd Blk Rate Avg"
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.drop_duplicates()
    return df



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
def prepare_for_logistic_regression(
    df,
    label_col="Label",
    fill_value=0
):
    df = df.copy()
    df.dropna(subset=[label_col], inplace=True)
    X = df.drop(columns=[label_col])
    y = df[label_col]


    y_encoded = y.apply(lambda v: 0 if str(v).strip().upper() == "BENIGN" else 1).values

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
            downcast="float"
        )


    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(fill_value, inplace=True)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    

    return X_scaled, y_encoded, scaler
