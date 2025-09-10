import pandas as pd
import numpy as np
import pickle
from pytorch_pre import build_horse_form_features, PreprocessArtifacts

def build_model_features_df(race_df: pd.DataFrame, artefacts_path="artifacts/preprocess.pkl") -> pd.DataFrame:
    """
    Given raw race_df (from schedule JSON or GUI table),
    produce a features DataFrame ready for model prediction.
    """
    with open(artefacts_path, "rb") as f:
        art: PreprocessArtifacts = pickle.load(f)

    df = race_df.copy()

    # Build horse form features if not already present
    if "horse_starts_prior" not in df.columns:
        form_feat = build_horse_form_features(df)
        df = pd.concat([df, form_feat], axis=1)

    # Ensure all required numeric/categorical cols are present
    for c in art.numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in art.categorical_cols:
        if c not in df.columns:
            df[c] = "UNK"

    # Impute numeric
    for c in art.numeric_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        df[c] = col.fillna(col.median())

    # Fill categoricals
    for c in art.categorical_cols:
        df[c] = df[c].astype(str).fillna("UNK").replace({"nan": "UNK"})

    return df[art.numeric_cols + art.categorical_cols]
