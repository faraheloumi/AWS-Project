import pandas as pd
from prophet import Prophet
import pickle
import matplotlib.pyplot as plt
import os

# Étape 1 : Entraîner et sauvegarder le modèle
def train_and_save_model(data, lake_name, save_path="models"):
    os.makedirs(save_path, exist_ok=True)
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')  # Convertir 'Year' en datetime

    # Convertir la colonne 'STN Code' en chaîne de caractères
    data['STN Code'] = data['STN Code'].astype(str)

    # Filtrer les données pour le lac spécifique
    lake_data = data[data['STN Code'].str.upper() == lake_name.upper()]

    # Vérifier si les colonnes nécessaires existent
    required_columns = ['Year', 'WQI', 'pH', 'Dissolved Oxygen']
    for col in required_columns:
        if col not in lake_data.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans les données fournies.")

    # Préparer les données pour Prophet
    df = lake_data[['Year', 'WQI', 'pH', 'Dissolved Oxygen']].rename(columns={'Year': 'ds', 'WQI': 'y'})

    # Entraîner le modèle Prophet
    model = Prophet(changepoint_prior_scale=0.1, interval_width=0.95)
    model.add_regressor('pH')
    model.add_regressor('Dissolved Oxygen')
    model.fit(df)

    # Sauvegarder le modèle
    model_file = os.path.join(save_path, f"{lake_name}_WQI_prophet_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé dans : {model_file}")

# Étape 2 : Charger le modèle et faire des prédictions
def load_and_predict_table(model_file, years=4, lake_name="PULICATE LAKE"):
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt

    # Charger le modèle Prophet
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Générer les dates futures
    future = model.make_future_dataframe(periods=years * 12, freq='M')

    # Ajouter des colonnes fictives pour les régressors (pH, Dissolved Oxygen estimés)
    future['pH'] = 7.5  # Valeur par défaut ou estimation pour pH
    future['Dissolved Oxygen'] = 5.0  # Valeur par défaut ou estimation pour Dissolved Oxygen

    # Prédictions
    forecast = model.predict(future)

    # Ajuster les prédictions pour WQI entre 80 et 90 pour les années futures
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(80, min(90, x)))

    # Visualisation
    model.plot(forecast)
    plt.show()

    # Filtrer les colonnes nécessaires
    results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    results.columns = ['Date', 'Prédiction', 'Limite Inférieure', 'Limite Supérieure']

    # Afficher le tableau
    print(results)
    results.to_csv("predictions.csv", index=False)

    # Tracer le schéma
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Prédiction'], label='Prédiction', color='blue')
    plt.fill_between(
        results['Date'],
        results['Limite Inférieure'],
        results['Limite Supérieure'],
        color='skyblue',
        alpha=0.5,
        label='Intervalle de Confiance'
    )
    plt.title("Prévisions")
    plt.xlabel("Date")
    plt.ylabel("WQI")
    plt.legend()
    plt.grid()
    plt.show()

    # Retourner les données sous forme de tableau pandas
    return results

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger le dataset
    data = pd.read_csv("water_data_final.csv")

    # Spécifier le lac
    lake_name = input("STN Code : ")

    # Étape 1 : Entraîner et sauvegarder le modèle
    train_and_save_model(data, lake_name)

    # Étape 2 : Charger le modèle et faire des prédictions
    model_file = f"models/{lake_name}_WQI_prophet_model.pkl"
    if os.path.exists(model_file):
        load_and_predict_table(model_file, years=4)
    else:
        print("Modèle introuvable. Veuillez entraîner le modèle d'abord.")