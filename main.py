import os
import pandas as pd
from config import *
from src.data_loader import load_data
from src.model import train_models
from src.evaluation import plot_roc, compute_combined, categorize
from src.interpretability import shap_analysis, shap_individual


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df, y, vectors = load_data(DATA_PATH)

    results = pd.DataFrame({
        "Patient_ID": df['Patient File No.'],
        "PCOS": y
    })

    for name, X in vectors.items():
        print(f"\nEvaluating {name}")

        model, rf, X_test, y_test = train_models(X, y)

        plot_roc(model, X_test, y_test, name, PLOTS_DIR)

        probs = (model.predict_proba(X)[:, 1] * 100).round(2)
        results[f"{name}_Risk"] = probs

        shap_analysis(rf, X, name, PLOTS_DIR)
        shap_individual(rf, X, name, PLOTS_DIR)

    compute_combined(results)

    results['Risk_Category'] = results['Combined_Risk'].apply(categorize)

    results.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved risk_scores.csv with categories")


if __name__ == "__main__":
    main()