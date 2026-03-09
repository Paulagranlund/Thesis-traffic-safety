import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from extract_excel_info import extract_text_and_labels
import sys
import os

EXCEL_PATH = "data/....xlsx"
#df = pd.read_excel(EXCEL_PATH)  # For example, contains ["SUMMARY", "LABEL"]
COLUMN_RENAME_MAP = {
    "UHELDSDATO": "accident_date",
    "UHELDSART": "report_category",
    "KODE_UHELDSSITUATION": "encoded_accident_situation",
    "UHELDSSITUATION": "accident_situation",
    "UHELDSTEKST": "police_narrative",
    "AAR": "year",
}

def extract_and_combine(folder_path):
    all_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            file_path = os.path.join(folder_path, file)

            # Load file
            xls = pd.ExcelFile(file_path)

            if len(xls.sheet_names) != 1:
                raise ValueError(f"{file} has more than one sheet")

            df = xls.parse(xls.sheet_names[0], header=2)

            # Rename columns
            df = df.rename(columns=COLUMN_RENAME_MAP)

            # Keep only needed columns
            df = df[list(COLUMN_RENAME_MAP.values())]

            # Optional: track source file
            df["source_file"] = file

            all_dfs.append(df)

    # Combine everything
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

df = extract_and_combine("data/data udtræk")

print(len(df))

texts = df["police_narrative"].astype(str).tolist()
df["encoded_accident_situation"] = pd.to_numeric(
    df["encoded_accident_situation"], errors="coerce"
)
df["main_situation_class"] = (df["encoded_accident_situation"] // 100).astype("Int64")
labels = df["main_situation_class"].tolist() # Labels are 0, 1, 2, 4, 5, 6, 9
df["has_narrative"] = df["police_narrative"].notna()
df_full = df.copy()

print(len(df_full))

df_text = df_full.loc[df_full["has_narrative"]].copy()
df_text["n_words"] = df_text["police_narrative"].str.split().str.len()
df = df_text[df_text["n_words"] >= 3].copy()

print(len(df_text))

print("Number of samples per label:")
print(df["main_situation_class"].value_counts().sort_index())
#df = pd.read_excel(EXCEL_PATH)  # For example, contains ["SUMMARY", "LABEL"]
#df = extract_text_and_labels("Our Analysis/data/2025 only.xlsx", sheet_number=0)
#texts = df["SUMMARY"].astype(str).tolist()
#labels = df["MANCOLL"].tolist()  # Labels are 0, 1, 2, 4, 5, 6, 9
#print("Number of samples per label:")
#print(df["MANCOLL"].value_counts().sort_index())

# 2. Load pre-trained BERT (no fine-tuning)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
#tokenizer = BertTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
#model = BertModel.from_pretrained("Maltehb/danish-bert-botxo")
model.eval()  # Evaluation mode, no parameter updates

# 3. Define feature extraction function
def get_bert_embeddings(texts, batch_size=16, max_len=128):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
            outputs = model(**inputs)
            # [CLS] vector
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.extend(cls_embeddings.numpy())
    return all_embeddings

# 4. Extract BERT vectors for SUMMARY
X = get_bert_embeddings(texts)

# 5. Split into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# 6. Use Logistic Regression for classification
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# 7. Testing & evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# Original evaluation on all labels
y_pred = clf.predict(X_test)
print("=== All labels (including 9) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro-F1:", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred, digits=3))

# Remove samples with label 9
mask = np.array(y_test) != 9
y_test_filtered = np.array(y_test)[mask]
y_pred_filtered = np.array(y_pred)[mask]

print("\n=== After removing label 9 ===")
print("Accuracy:", accuracy_score(y_test_filtered, y_pred_filtered))
print("Macro-F1:", f1_score(y_test_filtered, y_pred_filtered, average="macro"))
print(classification_report(y_test_filtered, y_pred_filtered, digits=3))

print("hejsa")