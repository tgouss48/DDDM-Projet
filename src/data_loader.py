"""
Chargement et audit des données d'incendies de forêts marocains.
Sources combinées : NASA FIRMS, NOAA GSOD, UN population, NDVI.
"""

import os
import pandas as pd
import numpy as np

def load_raw_data(data_dir="data"):
    """
    Charger le jeu de données
    """
    csv_path = os.path.join(data_dir, "Morocco_Wildfire_Predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier de données est manquant : {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def run_data_audit(df):
    """
    Effectuer un audit complet des données (complétude, cohérence, 
    distribution de la cible, doublons, valeurs manquantes).
    """
    audit_report = {}
    
    # 1. Dimensions du dataset
    audit_report['shape'] = df.shape
    
    # 2. Analyse des valeurs manquantes
    missing_series = df.isnull().sum()
    audit_report['missing_total'] = int(missing_series.sum())
    audit_report['missing_by_col'] = missing_series[missing_series > 0].to_dict()
    
    # 3. Analyse des doublons
    audit_report['duplicates'] = int(df.duplicated().sum())
    
    # 4. Distribution de la variable cible
    if 'is_fire' in df.columns:
        target_counts = df['is_fire'].value_counts().to_dict()
        audit_report['target_distribution'] = target_counts
        audit_report['fire_percentage'] = float(df['is_fire'].mean() * 100)
    else:
        audit_report['target_distribution'] = {}
        audit_report['fire_percentage'] = 0.0
        
    # 5. Statistiques clés sur les variables physiques
    physical_cols = ['NDVI', 'SoilMoisture', 'average_temperature_lag_1', 'precipitation_lag_1']
    stats = {}
    for col in physical_cols:
        if col in df.columns:
            stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }
    audit_report['physical_stats'] = stats
    
    return audit_report

def get_data_dictionary():
    """
    Retourner le dictionnaire de données documenté.
    """
    dictionary = [
        {"Column": "acq_date", "Type": "Date / Chaîne de caractères", "Source": "NASA FIRMS", 
         "Description": "Date d'acquisition des observations satellites."},
        {"Column": "latitude", "Type": "Flottant", "Source": "NASA FIRMS", 
         "Description": "Latitude de la zone d'observation au Maroc."},
        {"Column": "longitude", "Type": "Flottant", "Source": "NASA FIRMS", 
         "Description": "Longitude de la zone d'observation au Maroc."},
        {"Column": "NDVI", "Type": "Flottant", "Source": "MODIS (Vegetation Index)", 
         "Description": "Indice de végétation par différence normalisée (représente la densité de biomasse verte)."},
        {"Column": "SoilMoisture", "Type": "Flottant", "Source": "NASA/USDA", 
         "Description": "Humidité du sol (quantité d'eau retenue dans la couche supérieure du sol)."},
        {"Column": "sea_distance", "Type": "Flottant", "Source": "Calculé", 
         "Description": "Distance la plus proche par rapport à la mer/océan."},
        {"Column": "average_temperature_lag_1", "Type": "Flottant", "Source": "NOAA GSOD", 
         "Description": "Température moyenne observée la veille (lag 1) par la station météo la plus proche."},
        {"Column": "precipitation_lag_1", "Type": "Flottant", "Source": "NOAA GSOD", 
         "Description": "Précipitations observées la veille (lag 1) en millimètres."},
        {"Column": "wind_speed_lag_1", "Type": "Flottant", "Source": "NOAA GSOD", 
         "Description": "Vitesse du vent enregistrée la veille (lag 1)."},
        {"Column": "dew_point_lag_1", "Type": "Flottant", "Source": "NOAA GSOD", 
         "Description": "Point de rosée de la veille (lag 1) indiquant le niveau d'humidité de l'air."},
        {"Column": "is_fire", "Type": "Binaire (Entier / Flottant)", "Source": "NASA FIRMS", 
         "Description": "Variable cible (1.0 = Présence de feu actif détecté ; 0.0 = Pas d'incendie détecté)."}
    ]
    return pd.DataFrame(dictionary)

if __name__ == '__main__':
    try:
        df = load_raw_data()
        report = run_data_audit(df)
        print("--- Audit des Données de Wildfire Maroc ---")
        print(f"Dimensions : {report['shape']}")
        print(f"Nombre de doublons : {report['duplicates']}")
        print(f"Distribution de la cible is_fire : {report['target_distribution']}")
        print(f"Pourcentage d'incendies : {report['fire_percentage']:.2f}%")
        print("\n--- Quelques variables clés ---")
        for col, st in report['physical_stats'].items():
            print(f"{col} -> Min: {st['min']:.2f}, Max: {st['max']:.2f}, Mean: {st['mean']:.2f}")
    except Exception as e:
        print(f"Échec du test unitaire du chargeur de données : {e}")
