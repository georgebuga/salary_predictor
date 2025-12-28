import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import sys

# Load dataset
df = pd.read_csv("salary_data.csv")
print("📊 Dataset inițial:", df.shape)

# Select columns of interest
cols = ["Country", "EdLevel", "YearsCode", "Employment", "ConvertedCompYearly"]
df = df[cols].dropna()
print("✅ După dropna:", df.shape)

# Employment flexibil (nu strict 'Employed full-time')
df = df[df["Employment"].str.contains("Employed", na=False)]
print("📌 După filtrare Employment:", df.shape)

# Filtrare outliers
df = df[(df["ConvertedCompYearly"] < 500000) & (df["ConvertedCompYearly"] > 1000)]
print("📉 După filtrare salarii:", df.shape)

# Convert YearsCode
def clean_experience(x):
    if x == "More than 50 years":
        return 50
    if x == "Less than 1 year":
        return 0.5
    try:
        return float(x)
    except:
        return np.nan

df["YearsCode"] = df["YearsCode"].apply(clean_experience)
df = df.dropna()
print("📆 După curățare YearsCode:", df.shape)

# Dacă nu mai sunt date, oprește scriptul
if df.shape[0] < 10:
    print("❌ Prea puține date după filtrare! Verifică filtrările.")
    sys.exit(1)

# Encode categorice
le_country = LabelEncoder()
df["Country"] = le_country.fit_transform(df["Country"])

le_education = LabelEncoder()
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])

# Pregătire features/target
X = df[["Country", "EdLevel", "YearsCode"]]
y = df["ConvertedCompYearly"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧠 Train size: {len(X_train)}, Test size: {len(X_test)}")

# Antrenare model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Salvare model și encodere
with open("model/salary_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/le_country.pkl", "wb") as f:
    pickle.dump(le_country, f)

with open("model/le_education.pkl", "wb") as f:
    pickle.dump(le_education, f)

print("✅ Model salvat cu succes.")
