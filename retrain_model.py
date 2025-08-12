import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
csv_path = 'data/preprocess/Rockburst_in_Tunnel_V3.csv'
df = pd.read_csv(csv_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X = df.drop('Intensity_Level_encoded', axis=1)
y = df['Intensity_Level_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'artifacts/models/rockburst_rf_model.pkl')
joblib.dump(scaler, 'artifacts/models/feature_scaler.pkl')

print('Model and scaler retrained and saved.')
