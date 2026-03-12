import os
import random
import numpy as np
import pandas as pd
from typing import List
import time
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SEED = 42
MODEL_NAME = "bert-base-uncased" # choose pre-trained model, this is a bert model, trained on 110 million parameters (weights and biases), lowercased text
MAX_LENGTH = 256 # maximum number of tokens the model will process for each text input
OUTPUT_DIR = "Our Analysis/results/mancoll_bert2" # where to store the output; model weights, tokenizer, checkpoints, logs...
TEXT_COL = "SUMMARY" 
LABEL_COL = "MANCOLL"                 

# function to set all seeds, random, numpy, torch and torch.cuda
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ======== Load and Combine All Data Files ========

DATA_FOLDER = "data/data udtræk"

COLUMN_RENAME_MAP = {
    "UHELDSDATO": "accident_date",
    "UHELDSART": "report_category",
    "KODE_UHELDSSITUATION": "encoded_accident_situation",
    "UHELDSSITUATION": "accident_situation",
    "UHELDSTEKST": "police_narrative",
    "AAR": "year",
}

print("Loading all Excel files...")

all_dfs = []

for file in sorted(os.listdir(DATA_FOLDER)):
    if file.endswith(".xlsx"):
        path = os.path.join(DATA_FOLDER, file)
        print("Loading:", file)

        df_temp = pd.read_excel(path, header=2)

        # Rename columns
        df_temp = df_temp.rename(columns=COLUMN_RENAME_MAP)

        # Keep relevant columns
        df_temp = df_temp[list(COLUMN_RENAME_MAP.values())]

        all_dfs.append(df_temp)

# Combine all years
df = pd.concat(all_dfs, ignore_index=True)

print("Combined dataset shape:", df.shape)

# Convert encoded accident situation to numeric
df["encoded_accident_situation"] = pd.to_numeric(
    df["encoded_accident_situation"], errors="coerce"
)

# Create main class (e.g., 201 -> 2)
df["main_situation_class"] = (
    df["encoded_accident_situation"] // 100
).astype("Int64")

# Remove rows without narrative
df = df[df["police_narrative"].notna()].copy()

# Remove very short texts
df["n_words"] = df["police_narrative"].str.split().str.len()
df = df[df["n_words"] >= 3].copy()

print("After cleaning:", df.shape)
print("Samples per year:")
print(df["year"].value_counts().sort_index())

# ======== DEBUG: Run on small subset ========
DEBUG_SAMPLE_SIZE = 3000  # change to None to run full dataset

if DEBUG_SAMPLE_SIZE is not None:
    df = df.sample(n=DEBUG_SAMPLE_SIZE, random_state=SEED)
    print("Using debug subset:", df.shape)

# ======== Label Encoding ========

TEXT_COL = "police_narrative"
LABEL_COL = "main_situation_class"

unique_labels = sorted(int(x) for x in df[LABEL_COL].unique())

label2id = {orig: i for i, orig in enumerate(unique_labels)}
id2label = {i: orig for orig, i in label2id.items()}
num_labels = len(unique_labels)

# Apply encoding to entire dataset
df[LABEL_COL] = df[LABEL_COL].map(label2id).astype(int)

print("Label mapping:", label2id)

# ======== Stratified Train/Test Split ========

df_trainval, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df[LABEL_COL]
)

df_train, df_val = train_test_split(
    df_trainval,
    test_size=0.1,
    random_state=SEED,
    stratify=df_trainval[LABEL_COL]
)

print("Training samples:", len(df_train))
print("Validation samples:", len(df_val))
print("Test samples:", len(df_test))

# check distribution
print("\nClass distribution (train):")
print(df_train[LABEL_COL].value_counts(normalize=True).sort_index())

print("\nClass distribution (validation):")
print(df_val[LABEL_COL].value_counts(normalize=True).sort_index())

print("\nClass distribution (test):")
print(df_test[LABEL_COL].value_counts(normalize=True).sort_index())

# ======== Tokenizer and Dataset ========
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # loads bert's tokenizer

# converts the text to tokenized text and labels to tensor. 
# Ex. 
# input: text: "Car hit the rear of truck", label: 0. 
# output: input_ids: tensor([101, 2487, 2718, 1996, 4396, 1997, 4744, 102]) where 101 and 102 are start and end tokens, attention_mask: tensor([1, 1, 1, 1, 1, 1, 1, 1]), labels: tensor(0)
class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str, label_col: str, tokenizer, max_length: int):
        self.texts = df[text_col].tolist() # all text to a list
        self.labels = df[label_col].tolist() # all labels to a list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts) # count number of samples

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        # tokenize the text (converts to input ids (numbers instead of words, one id for each token) and attention mask (tells if a word is real or fake for padding) and pads to max length)
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()} # format it correctly
        item["labels"] = torch.tensor(label, dtype=torch.long) # converts label to tensor: 3 -> tensor(3)
        return item

# use the class to convert train, test and validation sets, its here where they mute 2021 test set and use 2020 as test instead
train_ds = TextClsDataset(df_train, TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
val_ds   = TextClsDataset(df_val,   TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)
test_ds = TextClsDataset(df_test, TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH)

# ======== Dealing with class imbalance (class weights) ========
class_counts = df_train[LABEL_COL].value_counts().sort_index().values # count number of samples in each class
class_weights = (1.0 / (class_counts + 1e-9)) # compute inverse frequency weights; rare classes get higher weights
class_weights = class_weights / class_weights.sum() * num_labels # normalizes weights
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float) # convert NumPy arrays into a PyTorch tensors

# Customize Trainer to inject class weights
# takes the normal Trainer and modifies how the loss is calculated so we can use class weights 
# rare classes get punished more and common classes get punished less
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # if class weights are given, move them to same device as the model
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.model.device)
        else:
            self.class_weights = None

    # modify the default loss calculation
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels") # extract true labels
        outputs = model(**inputs) # run model
        logits = outputs.get("logits") # predicts raw scores (logits), not yet probabilities, ex: [2.3, -1.1, 0.5]

        # choose loss function, if weights exist -> use weighted loss, if not -> use normal loss
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = CrossEntropyLoss()

        # calculate loss
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

# ======== Model ========
# load a pre-trained BERT model, add a classification head and prepare it for text classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,            # name or path of the pre-trained model
    num_labels=num_labels, # number of output classes
    id2label=id2label,     # mapping from class index -> original label value
    label2id=label2id,     # reverse the dictionary, mapping from original label value -> class index
)

# ======== Evaluation metrics ========
def compute_metrics(eval_pred):
    logits, labels = eval_pred # logits: raw outputs from the model, labels: true labels
    preds = np.argmax(logits, axis=1)  # take the index highest scoring class
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")  # equal weight for each class
    return {"accuracy": acc, "f1_macro": f1_macro}

# ======== Training arguments ========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # where the model checkpoints will be saved
    per_device_train_batch_size=16, # how many samples the model processes at once during training
    per_device_eval_batch_size=32, # how many samples the model processes at once during test/validation
    learning_rate=2e-5, # how big the update steps are for each batch
    num_train_epochs=5, # how many times the model sees the entire training dataset
    weight_decay=0.01, # regularization, penalty on big weights, prevents overfitting
    eval_strategy="epoch", # when to evaluate, here after each epoch
    save_strategy="epoch", # save a checkpoint after each epoch.
    load_best_model_at_end=True, # after training, the model with the best validation score will be loaded, not necessarily the last epoch
    metric_for_best_model="f1_macro", # choose the best model based on macro F1, not accuracy, good choice for class imbalance
    greater_is_better=True, # higher F1 is better
    logging_steps=50, # print training logs every 50 steps
    save_total_limit=5, # keep only the 5 most recent checkpoints
    report_to="none",
)

# use the weighted trainer created earlier
trainer = WeightedTrainer(
    model=model, # use bert model defined earlier
    args=training_args, # use training arguments defined above
    train_dataset=train_ds, # use training set from 2021
    eval_dataset=val_ds, # use validation set from 2021 to evaluate
    tokenizer=tokenizer,
    data_collator=default_data_collator, # takes individual tokenized samples and combines them into stacked batches
    compute_metrics=compute_metrics, # use compute metrics function defined earlier
)

# ======== training ========
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

from transformers import AutoModelForSequenceClassification

def eval_and_print(name, dataset):
    if dataset is None:
        return

    print(f"\n== {name} ==")

    time_start = time.time()
    preds = trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro'):.4f}")

    print(classification_report(y_true, y_pred))

    time_end = time.time()
    print("Runtime (seconds):", time_end - time_start)

# eval_and_print("Validation", val_ds)
eval_and_print("Test", test_ds) # evaluate the trained model on the test dataset and print the results

# function that predicts labels for new texts after the model has been trained
def predict_texts(texts: List[str]) -> List[int]: # input is a list of strings (texts) and output is a list of integers (predicted labels)
    # convert words into input IDs: truncate if longer than MAX_LENGTH, pad shorter texts, return PyTorch tensors
    enc = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(trainer.model.device) for k, v in enc.items()} # move tensors to correct device so that model and data are on the same device
    with torch.no_grad(): # disable gradient tracking, as we are only predicting, not training
        logits = trainer.model(**enc).logits # feeds the tokenized texts into the model and outputs raw scores (logits) for each class
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist() # selects the class with the highest logit and converts tensor to CPU to NumPy to Python list
    return [id2label[p] for p in preds] # convert back to original labels