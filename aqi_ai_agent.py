import pandas as pd
import numpy as np
import os
import time
import joblib
import folium
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import gradio as gr

# === Load & Clean Dataset ===
def load_data(file_path):
    df = pd.read_csv(file_path)
    pollutant_cols = ['2017 - PM2.5', '2017 - PM10', '2017 - SO2', '2017 - NO2']
    df[pollutant_cols] = df[pollutant_cols].replace(['-', '@'], np.nan).astype(float)
    df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].mean())
    return df, pollutant_cols

# === Train Model ===
def train_model(df, pollutant_cols):
    X = df[pollutant_cols]
    y = df['2017 - PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    print("Model trained ✅")
    print(f"Mean Squared Error: {mse:.2f} | R² Score: {r2:.2f}")
    return model

# === Geocode Cities ===
def geocode_cities(df):
    geolocator = Nominatim(user_agent="air_quality_mapper", timeout=10)
    def get_coordinates(city):
        try:
            location = geolocator.geocode(city + ", India")
            time.sleep(1)
            return pd.Series([location.latitude, location.longitude])
        except:
            return pd.Series([None, None])
    df[['latitude', 'longitude']] = df['Cities'].apply(get_coordinates)
    return df.dropna(subset=['latitude', 'longitude'])

# === Plotly Map Generator ===
def generate_map(df):
    fig = go.Figure(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=9, color='red'),
        text=df['Cities'] + ": AQI " + df['2017 - PM2.5'].astype(str)
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 22.5, "lon": 78.9},
        mapbox_zoom=4,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig

# === AQI Advisory ===
def health_warning(pm25):
    if pm25 <= 50:
        return "✅ Good: Air quality is satisfactory."
    elif pm25 <= 100:
        return "🟡 Moderate: Sensitive individuals should reduce outdoor activity."
    elif pm25 <= 200:
        return "🟠 Unhealthy: Wear masks and avoid outdoor exertion."
    else:
        return "🔴 Very Unhealthy: Stay indoors and consult a doctor if symptoms appear."

# === City Pollutant Report ===
def city_report(city):
    row = df[df['Cities'] == city].iloc[0]
    report = f"""
    **City:** {city}  
    **PM2.5:** {row['2017 - PM2.5']} µg/m³  
    **PM10:** {row['2017 - PM10']} µg/m³  
    **SO₂:** {row['2017 - SO2']} µg/m³  
    **NO₂:** {row['2017 - NO2']} µg/m³  
    """
    return report + "\n\n" + health_warning(row['2017 - PM2.5'])

# === Predict AQI ===
def predict_aqi(pm25, pm10, so2, no2):
    prediction = model.predict([[pm25, pm10, so2, no2]])[0]
    return f"Predicted AQI: {round(prediction, 2)}\n{health_warning(prediction)}"

# === Show Map Button Handler ===
def show_map():
    return generate_map(df_mapped)

# === Run Everything ===
if __name__ == "__main__":
    file_path = "Air Quality in INDIA.csv"
    df, pollutant_cols = load_data(file_path)
    model = train_model(df, pollutant_cols)
    df_mapped = geocode_cities(df)

    with gr.Blocks() as app:
        gr.Markdown("## 🌍 Indian Air Quality Dashboard")

        with gr.Tab("📍 City AQI Report"):
            city = gr.Dropdown(choices=df['Cities'].tolist(), label="Select a City")
            output = gr.Markdown()
            city.change(fn=city_report, inputs=city, outputs=output)

        with gr.Tab("🔮 Predict AQI"):
            pm25 = gr.Slider(0, 500, label="PM2.5")
            pm10 = gr.Slider(0, 500, label="PM10")
            so2 = gr.Slider(0, 100, label="SO₂")
            no2 = gr.Slider(0, 100, label="NO₂")
            result = gr.Textbox()
            gr.Button("Predict").click(predict_aqi, inputs=[pm25, pm10, so2, no2], outputs=result)

        with gr.Tab("🗺️ Pollution Map"):
            gr.Markdown("Click to load AQI markers on map")
            map_plot = gr.Plot()
            gr.Button("Load Map").click(fn=show_map, inputs=[], outputs=map_plot)

    app.launch()
