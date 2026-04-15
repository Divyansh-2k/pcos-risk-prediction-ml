import shap
import matplotlib.pyplot as plt
import os


def shap_analysis(model, X, name, save_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(save_dir, f"shap_{name}.png"), bbox_inches='tight')
    plt.close()


def shap_individual(model, X, name, save_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    vals = shap_values[1][0]

    exp = shap.Explanation(
        values=vals,
        base_values=explainer.expected_value[1],
        data=X.iloc[0].values,
        feature_names=X.columns.tolist()
    )

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(exp, show=False)
    plt.savefig(os.path.join(save_dir, f"waterfall_{name}.png"), bbox_inches='tight')
    plt.close()