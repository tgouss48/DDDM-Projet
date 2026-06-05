"""
Pipeline d'entraînement des modèles prédictifs, clustering des zones de risque et explicabilité SHAP.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ajouter le répertoire racine au PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_data, run_data_audit
from src.preprocessor import clean_data, perform_feature_engineering, prepare_ml_datasets
from src.models import train_and_evaluate_models, calculate_shap_explainer, save_model_artifacts

def main():
    print("==================================================")
    print("DEMARRAGE DU PIPELINE - INCENDIES DE FORETS MAROC")
    print("==================================================")
    
    # 1. Chargement des données
    print("[1/6] Chargement des données d'incendies...")
    try:
        raw_df = load_raw_data("data")
        print(f"Jeu de données chargé avec succès : {len(raw_df)} lignes.")
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
        
    # 2. Audit des données
    print("[2/6] Exécution de l'audit de qualité des données...")
    audit_report = run_data_audit(raw_df)
    print(f"Audit terminé ! Taux d'incendies dans le dataset : {audit_report['fire_percentage']:.2f}%.")
    
    # 3. Nettoyage et Feature Engineering
    print("[3/6] Nettoyage et feature engineering...")
    cleaned_df = clean_data(raw_df)
    engineered_df = perform_feature_engineering(cleaned_df)
    print("Feature engineering terminé. Variables créées avec succès.")
    
    # 4. Clustering pour segmentation des zones de risques
    print("[4/6] Segmentation des zones de risques par K-Means...")
    # On segmente par latitude, longitude et FWI (Fire Weather Index) moyen
    kmeans_features = ['latitude', 'longitude', 'fwi_index']
    scaler_km = StandardScaler()
    km_scaled = scaler_km.fit_transform(engineered_df[kmeans_features])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    engineered_df['Risk_Cluster'] = kmeans.fit_predict(km_scaled)
    
    # Identifier les profils des clusters de risques
    cluster_profiles = engineered_df.groupby('Risk_Cluster').agg({
        'is_fire': 'mean',
        'temp_max_c': 'mean',
        'wind_speed_lag_1': 'mean',
        'NDVI': 'mean',
        'latitude': 'count'
    }).rename(columns={'latitude': 'ObservationCount'}).reset_index()
    
    # Trier les clusters par taux de feu moyen pour les labelliser intuitivement
    cluster_profiles = cluster_profiles.sort_values(by='is_fire').reset_index(drop=True)
    # Associer un niveau de risque textuel
    risk_labels = {0: 'Risque Faible', 1: 'Risque Modéré', 2: 'Risque Élevé'}
    cluster_profiles['Risk_Label'] = cluster_profiles.index.map(risk_labels)
    
    # Mettre à jour dans le dataframe principal
    cluster_mapping = dict(zip(cluster_profiles['Risk_Cluster'], cluster_profiles['Risk_Label']))
    engineered_df['Risk_Segment'] = engineered_df['Risk_Cluster'].map(cluster_mapping)
    
    print("Profils des clusters de risques créés :")
    print(cluster_profiles)
    
    # 5. Modélisation Prédictive (Classification is_fire)
    print("[5/6] Entraînement des classifieurs...")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_ml_datasets(engineered_df)
    report, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("\nRapport de performance des modèles :")
    print("--------------------------------------------------")
    print(report.to_string(index=False))
    print("--------------------------------------------------")
    
    # On choisit le meilleur modèle en fonction du score F1 ou Recall
    # Le rappel (Recall) est très important pour la protection civile, mais on veut un bon équilibre (F1-Score)
    best_model_row = report.sort_values(by='F1-Score', ascending=False).iloc[0]
    best_model_name = best_model_row['Model']
    best_model = models[best_model_name]
    print(f"Meilleur modèle retenu : {best_model_name} (F1: {best_model_row['F1-Score']:.4f}, Recall: {best_model_row['Recall']:.4f})")
    
    # 6. Explicabilité SHAP et sauvegarde des artefacts
    print("[6/6] Calcul de l'explicabilité SHAP globale/locale...")
    explainer, shap_values = calculate_shap_explainer(best_model, X_train, X_test)
    
    # Compiler les artefacts
    artifacts = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'models_report': report,
        'explainer_serialized': explainer,
        'shap_values_serialized': shap_values,
        'engineered_df': engineered_df.head(5000),  # Échantillon pour l'affichage fluide dans le dashboard
        'cluster_profiles': cluster_profiles,
        'audit_report': audit_report
    }
    
    # Sauvegarde
    save_model_artifacts(artifacts, output_dir="data")
    print("==================================================")
    print("Pipeline terminé ! Lancez Streamlit pour afficher le tableau de bord.")
    print("==================================================")

if __name__ == '__main__':
    main()
