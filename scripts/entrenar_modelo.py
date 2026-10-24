"""Entrena el modelo de producción y lo guarda en models/modelo_score_cobranzas.pkl.
Uso:  python scripts/entrenar_modelo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelo as sc  # noqa: E402

if __name__ == "__main__":
    mdl, met = sc.entrenar()
    sc.guardar(mdl, met)
    print(f"AUC prueba = {met['auc']:.4f} · KS = {met['ks']:.4f} · entrenamiento {met['n_train']} / prueba {met['n_test']}")
    print("guardado:", sc.MODELO_PKL)
