"""
Feature engineering et prétraitement pour la prédiction des incendies.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Nettoyage initial des données et traitement des valeurs manquantes.
    """
    df = df.copy()
    
    # 1. Traitement des valeurs manquantes (imputation simple par médiane/mode)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # 2. S'assurer que is_fire est un entier
    if 'is_fire' in df.columns:
        df['is_fire'] = df['is_fire'].astype(int)
        
    return df

def perform_feature_engineering(df):
    """
    Génèrer de nouvelles variables explicatives pertinentes pour le risque d'incendie.
    """
    df = df.copy()
    
    # Convertir acq_date en datetime
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['Month'] = df['acq_date'].dt.month
    df['Year'] = df['acq_date'].dt.year
    
    # Saison des feux au Maroc (Juin à Septembre)
    df['is_fire_season'] = df['Month'].isin([6, 7, 8, 9]).astype(int)
    
    # Conversion de Fahrenheit en Celsius pour les températures
    # La formule : (F - 32) * 5/9
    df['temp_avg_c'] = (df['average_temperature_lag_1'] - 32) * 5.0 / 9.0
    df['temp_max_c'] = (df['maximum_temperature_lag_1'] - 32) * 5.0 / 9.0
    df['temp_min_c'] = (df['minimum_temperature_lag_1'] - 32) * 5.0 / 9.0
    
    # Indice météo simplifié du risque d'incendie (Fire Weather Index - FWI)
    # Plus la température est élevée, le vent fort, l'humidité du sol basse et les précipitations nulles,
    # plus le risque d'incendie est élevé.
    df['fwi_index'] = (
        (df['temp_max_c'] * 0.4) + 
        (df['wind_speed_lag_1'] * 0.3) - 
        (df['precipitation_lag_1'] * 0.5) + 
        (100 - df['SoilMoisture']) * 0.2
    )
    
    # labellisation des régions du Maroc par rapport à la latitude
    # Nord : > 34.0 (Tanger-Tétouan, etc.)
    # Centre : Entre 31.5 et 34.0 (Rabat, Casa, Fès, Marrakech)
    # Sud : < 31.5 (Agadir, Provinces du Sud)
    def classify_region(lat):
        if lat > 34.0:
            return 'Nord'
        elif lat >= 31.5:
            return 'Centre'
        else:
            return 'Sud'
            
    df['Region'] = df['latitude'].apply(classify_region)
    
    # One-hot encoding de la variable Region
    region_dummies = pd.get_dummies(df['Region'], prefix='Region').astype(int)
    df = pd.concat([df, region_dummies], axis=1)
    
    return df

def prepare_ml_datasets(df, test_size=0.2, random_state=42):
    """
    Sélectionne les features, sépare en train/test et normalise les variables.
    """
    # 1. Sélection finale des variables explicatives
    features = [
        'latitude', 'longitude', 'NDVI', 'SoilMoisture', 'sea_distance',
        'is_holiday', 'day_of_week', 'day_of_year', 'is_weekend',
        'temp_avg_c', 'temp_max_c', 'temp_min_c', 'precipitation_lag_1',
        'wind_speed_lag_1', 'dew_point_lag_1', 'fwi_index', 'is_fire_season',
        'Region_Nord', 'Region_Centre', 'Region_Sud'
    ]
    
    # Assurer que toutes les colonnes requises existent
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['is_fire']
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, features

if __name__ == '__main__':
    from data_loader import load_raw_data
    try:
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        engineered_df = perform_feature_engineering(cleaned_df)
        X_train, X_test, y_train, y_test, scaler, features = prepare_ml_datasets(engineered_df)
        
        print("--- Test Unitaire du Préprocesseur ---")
        print(f"Features sélectionnées ({len(features)}) : {features}")
        print(f"Taille X_train : {X_train.shape}")
        print(f"Taille X_test : {X_test.shape}")
        print(f"Moyenne FWI Index : {engineered_df['fwi_index'].mean():.2f}")
        print(f"Distribution des régions :\n{engineered_df['Region'].value_counts()}")
    except Exception as e:
        print(f"Échec du test unitaire du préprocesseur : {e}")
