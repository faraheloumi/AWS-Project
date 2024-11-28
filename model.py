import pandas as pd
from prophet import Prophet
import os
import json

def forecast_parameters(data, start_year=2017, end_forecast_year=2026):
    # Conversion de la colonne 'Year' au format datetime
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    
    # Créez un répertoire pour les résultats
    base_directory = "forecast_results"
    if not os.path.exists(base_directory):
        os.mkdir(base_directory)
    
    # Liste des années à traiter
    years = list(range(start_year, end_forecast_year + 1))
    
    for year in years:
        year_directory = os.path.join(base_directory, str(year))
        if not os.path.exists(year_directory):
            os.mkdir(year_directory)
        
        for stn_code in data['STN Code'].unique():
            stn_data = data[data['STN Code'] == stn_code]
            
            if stn_data.empty:
                continue
            
            parameters = {}
            for column in ['WQI', 'Dissolved Oxygen', 'pH', 'Conductivity', 'BOD', 
                           'Nitrate N + Nitrite N', 'Fecal Coliform', 'Total Coliform']:
                
                if year <= 2022:  # Utiliser les données existantes pour les années <= 2022
                    existing_value = stn_data[stn_data['Year'].dt.year == year]
                    parameters[column] = (
                        existing_value[column].values[0] if not existing_value.empty else None
                    )
                elif year <= end_forecast_year:  # Prévision pour les années > 2022
                    stn_data_param = stn_data.rename(columns={'Year': 'ds', column: 'y'})
                    stn_data_param = stn_data_param[['ds', 'y']].dropna()

                    if len(stn_data_param) >= 2:  # Au moins 2 points nécessaires pour Prophet
                        model = Prophet(changepoint_prior_scale=0.001, yearly_seasonality=True)
                        model.fit(stn_data_param)

                        future = model.make_future_dataframe(periods=(end_forecast_year - 2022), freq='Y')
                        forecast = model.predict(future)

                        forecast_value = forecast.loc[forecast['ds'].dt.year == year, 'yhat']
                        parameters[column] = (
                            forecast_value.values[0] if not forecast_value.empty else None
                        )
                    else:
                        parameters[column] = None

            # Ajouter les informations statiques
            stn_details = stn_data.iloc[0].to_dict()
            parameters.update({
                "Location Name": stn_details.get("Location Name"),
                "State Name": stn_details.get("State Name"),
                "lat": stn_details.get("lat"),
                "lon": stn_details.get("lon"),
                "processed_timestamp": pd.Timestamp.now().isoformat()
            })

            # Enregistrer les résultats dans un fichier JSON
            stn_directory = os.path.join(year_directory, f"STN_{stn_code}")
            if not os.path.exists(stn_directory):
                os.mkdir(stn_directory)

            json_path = os.path.join(stn_directory, f"parameters_{year}.json")
            with open(json_path, 'w') as json_file:
                json.dump(parameters, json_file, indent=4)

# Charger les données
data_cleaned = pd.read_csv("water_data_final.csv")

# Appeler la fonction
forecast_parameters(data_cleaned)

print("Les résultats des prévisions ont été enregistrés dans le dossier 'forecast_results'.")