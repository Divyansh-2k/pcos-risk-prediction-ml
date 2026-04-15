from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def train_models(X, y):
    rf = RandomForestClassifier(n_estimators=150, max_depth=10,
                                class_weight='balanced', random_state=42)

    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('ada', ada)],
        voting='soft'
    )

    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "Ensemble_RF_Ada": ensemble
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():
        auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        print(f"{name}: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return best_model, rf, X_test, y_test