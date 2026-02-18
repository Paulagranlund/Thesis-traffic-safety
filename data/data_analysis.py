import pandas as pd

from extract_excel_info.py import extract_text_and_labels

# EXCEL_PATH = "data/....xlsx"
#df = pd.read_excel(EXCEL_PATH)  # For example, contains ["SUMMARY", "LABEL"]
df = extract_text_and_labels("data/2025 only.xlsx", sheet_number=0)
texts = df["SUMMARY"].astype(str).tolist()
labels = df["MANCOLL"].tolist()  # Labels are 0, 1, 2, 4, 5, 6, 9
df.head()


#priliminary analysis of label distribution
label_counts = df['MANCOLL'].value_counts()
print("Label Distribution:")
print(label_counts)

