@echo off
title DDDM Decision Support System Launcher - Incendies Maroc
color 0A

echo ======================================================================
echo          DDDM Decision Support System - Incendies de Forets Maroc
echo ======================================================================
echo.
echo Ce script va initialiser l'environnement, entrainer les modeles
echo de Machine Learning (XGBoost, Random Forest, Regression Logistique),
echo et lancer le tableau de bord Streamlit.
echo.
echo ----------------------------------------------------------------------
echo Etape 1 : Installation des dependances Python...
echo ----------------------------------------------------------------------
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo AVERTISSEMENT : L'installation de certaines dependances a echoue ou est deja satisfaite.
    echo Tentative de poursuite du script...
)

echo.
echo ----------------------------------------------------------------------
echo Etape 2 : Entrainement des modeles de prediction & SHAP Explainer...
echo ----------------------------------------------------------------------
python src/train_models.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERREUR : Echec de l'entrainement de la pipeline de Machine Learning.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ----------------------------------------------------------------------
echo Etape 3 : Lancement du Decision Dashboard (Streamlit)...
echo ----------------------------------------------------------------------
echo Le dashboard va s'ouvrir automatiquement dans votre navigateur par defaut.
echo (Vous pouvez fermer la fenetre en pressant Ctrl+C dans ce terminal)
echo.
streamlit run dashboard/app.py

pause
