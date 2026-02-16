import pandas as pd
EXCEL_PATH = "US Analysis/data/case_info_2020.xlsx"
df = pd.read_excel(EXCEL_PATH)  # For example, contains ["SUMMARY", "LABEL"]
texts = df["SUMMARY"].astype(str).tolist()
labels = df["MANCOLL"].tolist()  # Labels are 0, 1, 2, 4, 5, 6, 9
print("Number of samples per label:")
print(df["MANCOLL"].value_counts().sort_index())