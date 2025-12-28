import pandas as pd

df = pd.read_csv("salary_data.csv")

# Afișează primele 5 rânduri doar cu coloanele de interes
cols = ["Country", "EdLevel", "YearsCode", "Employment", "ConvertedCompYearly"]

missing_cols = [col for col in cols if col not in df.columns]
if missing_cols:
    print(f"Lipsesc coloanele: {missing_cols}")
else:
    print("✅ Coloanele există. Primele 5 rânduri:")
    print(df[cols].dropna().head())
