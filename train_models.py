#!/usr/bin/env python
"""
Treina dois modelos:
  • severity_clf.joblib  – classificador  (classes 1‒4)
  • severity_reg.joblib  – regressor      (valor contínuo de severidade)

Uso
----
$ python train_severity_models.py           # usa caminhos padrão
$ python train_severity_models.py --csv data/meus_dados.csv --out models/
$ python train_severity_models.py -h        # ajuda
"""
from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ─────────────────────────────────────────────────────────────────────────────
# Função de treino “pura”
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save(
    csv_path: str | Path,
    out_dir: str | Path = "models",
    features: list[str] | None = None,
    seed: int = 42,
) -> None:
    """
    Lê CSV, treina dois modelos (clf/reg) e salva em `out_dir`.

    Parameters
    ----------
    csv_path : str | Path
        Caminho do CSV contendo colunas de vitais + `severity_class` e
        `severity_value`.
    out_dir : str | Path, default="models"
        Pasta onde serão salvos os arquivos `.joblib`.
    features : list[str] | None, default=None
        Quais colunas usar como X.  Se None, usa
        ["pSist", "pDiast", "qPA", "pulse", "resp_freq"].
    seed : int, default=42
        Semente para reprodutibilidade.
    """
    csv_path = Path(csv_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if features is None:
        features = ["pSist", "pDiast", "qPA", "pulse", "resp_freq"]

    # 1) Carrega dados
    df = pd.read_csv(csv_path)

    X = df[features].values
    y_class = df["severity_class"].values
    y_value = df["severity_value"].values

    # 2) Pré-processamento (todas numéricas)
    preproc = ColumnTransformer(
        transformers=[("num", StandardScaler(), list(range(X.shape[1])))],
        remainder="drop",
    )

    # 3) Pipelines
    clf = Pipeline(
        steps=[
            ("pre", preproc),
            ("model", RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=seed)),
        ]
    )
    reg = Pipeline(
        steps=[
            ("pre", preproc),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=seed)),
        ]
    )

    # 4) Treina
    clf.fit(X, y_class)
    reg.fit(X, y_value)

    # 5) Salva
    clf_path = out_dir / "severity_clf.joblib"
    reg_path = out_dir / "severity_reg.joblib"
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    print(f"✅ Modelos gravados em: {clf_path}, {reg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Interface de linha de comando
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina e salva os modelos de severidade.")
    parser.add_argument("--csv",  "-c", default="training_data/training_vitals.csv",
                        help="Caminho do CSV de treino (default: %(default)s)")
    parser.add_argument("--out",  "-o", default="models",
                        help="Diretório de saída (default: %(default)s)")
    args = parser.parse_args()

    train_and_save(args.csv, args.out)
