"""
Entraînement, évaluation et explicabilité (SHAP) des modèles prédictifs.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Entraîner 3 modèles (Régression Logistique, Random Forest, XGBoost)
    et les évaluer selon plusieurs métriques, en optimisant le Recall.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models_report = []
    trained_models = {}
    
    # 1. Régression Logistique
    print("Entraînement de la Régression Logistique...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {'C': [0.1, 1.0, 10.0]}
    lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='recall', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    trained_models['Régression Logistique'] = best_lr
    
    # 2. Random Forest
    print("Entraînement de la Forêt Aléatoire...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100],
        'max_depth': [5, 10]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='recall', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    trained_models['Random Forest'] = best_rf
    
    # 3. XGBoost / Gradient Boosting
    if XGB_AVAILABLE:
        model_name = 'XGBoost'
        print("Entraînement de XGBoost...")
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model_name = 'Gradient Boosting'
        print("Entraînement du Gradient Boosting...")
        xgb = GradientBoostingClassifier(random_state=42)
        
    xgb_params = {
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=cv, scoring='recall', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    trained_models[model_name] = best_xgb
    
    # Évaluation comparée
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        models_report.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })
        
    return pd.DataFrame(models_report), trained_models

def calculate_shap_explainer(best_model, X_train, X_test):
    """
    Calculer l'explainer et les valeurs SHAP pour l'explicabilité globale et locale.
    """
    if not SHAP_AVAILABLE:
        print("SHAP n'est pas installé. Ignorer le calcul.")
        return None, None
        
    print("Calcul de SHAP...")
    try:
        model_type = type(best_model).__name__
        background = shap.kmeans(X_train, 100) if len(X_train) > 100 else X_train
        
        if 'RandomForest' in model_type or 'XGB' in model_type or 'GradientBoosting' in model_type:
            explainer = shap.TreeExplainer(best_model)
            try:
                shap_values = explainer(X_test.head(200))
            except Exception:
                explainer = shap.Explainer(best_model, background)
                shap_values = explainer(X_test.head(200))
        else:
            explainer = shap.Explainer(best_model, background)
            shap_values = explainer(X_test.head(200))
            
        return explainer, shap_values
    except Exception as e:
        print(f"Erreur calcul SHAP : {e}")
        return None, None

def save_model_artifacts(artifacts, output_dir="data"):
    """
    Enregistrer les artefacts (modèles, scaler, colonnes) au format pickle.
    """
    os.makedirs(output_dir, exist_ok=True)
    artifacts_path = os.path.join(output_dir, "model_artifacts.pkl")
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Artefacts sauvegardés avec succès dans {artifacts_path} !")

def load_model_artifacts(output_dir="data"):
    """
    Charger les artefacts.
    """
    artifacts_path = os.path.join(output_dir, "model_artifacts.pkl")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Les artefacts à {artifacts_path} sont introuvables.")
    with open(artifacts_path, 'rb') as f:
        return pickle.load(f)
