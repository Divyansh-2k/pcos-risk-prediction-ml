import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


def plot_roc(model, X_test, y_test, name, save_dir):
    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(save_dir, f"roc_{name}.png"))
    plt.close()


def compute_combined(df):
    df['Combined_Risk'] = (
        0.35 * df['Ovarian_Risk'] +
        0.30 * df['Hormonal_Risk'] +
        0.20 * df['Clinical_Risk'] +
        0.15 * df['Lifestyle_Risk']
    ).clip(0, 100)


def categorize(score):
    if score < 25:
        return "Low Risk"
    elif score < 60:
        return "Moderate Risk"
    return "High Risk"