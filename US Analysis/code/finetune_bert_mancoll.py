import os
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import time
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
OUTPUT_DIR = "./mancoll_bert2" # where to store the output; model weights, tokenizer, checkpoints, logs...
EXCEL_PATH = "US Analysis/data/case_info_2021.xlsx" # training data
test_path= "US Analysis/data/case_info_2020.xlsx" # test data
TEXT_COL = "SUMMARY" 
LABEL_COL = "MANCOLL"
VAL_SIZE = 0.1 # validation size, used on 2021        
TEST_SIZE = 0.1 # test size on 2021 (not used?)              

# function to set all seeds, random, numpy, torch and torch.cuda
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# load data
df = pd.read_excel(EXCEL_PATH)
df_test_extra = pd.read_excel(test_path)

# choose only text and labels, drop rows with missing values and reset index
df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True) 
df_test_extra = df_test_extra[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)

# ======== Label Encoding ========
# Original Category Set
unique_labels = sorted(int(x) for x in df[LABEL_COL].unique()) # create a sorted list of all unique label values, converted to integers
label2id = {orig: i for i, orig in enumerate(unique_labels)} # build a dictionary with indexes for the original label value (needed for neural networks: 0, 1, 2, ... instead of 5, 25, 12, 45, ...)
id2label = {i: orig for orig, i in label2id.items()} # reverse the dictionary
num_labels = len(unique_labels) # number of classes

# replace labels with the new labels from the dictionary
df[LABEL_COL] = df[LABEL_COL].map(label2id) 
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].map(label2id)

# ensure all labels exist in the mapping
if df[LABEL_COL].isnull().any():
    raise ValueError("Some labels could not be mapped! Check your LABEL_COL values.")
if df_test_extra[LABEL_COL].isnull().any():
    raise ValueError("Some labels in test set could not be mapped! Check your LABEL_COL values.")

# ensure labels are integers
df[LABEL_COL] = df[LABEL_COL].astype(int)
df_test_extra[LABEL_COL] = df_test_extra[LABEL_COL].astype(int)

# create a test set from 2021 data (which is not used later?)
if TEST_SIZE > 0:
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[LABEL_COL]
    )
else:
    df_trainval, df_test = df, None

# create training and validation sets from 2021
df_train, df_val = train_test_split(
    df_trainval, test_size=VAL_SIZE, random_state=SEED, stratify=df_trainval[LABEL_COL]
)

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
# test_ds  = TextClsDataset(df_test,  TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH) if df_test is not None else None
test_ds  = TextClsDataset(df_test_extra,  TEXT_COL, LABEL_COL, tokenizer, MAX_LENGTH) if df_test is not None else None

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
# trainer.train()
# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

from transformers import AutoModelForSequenceClassification

def eval_and_print(name, dataset): # input: name, a label like "Test" or "Validation" and dataset, the dataset we want to evaluate on
    if dataset is None:
        return
# Reload the model from the saved checkpoint, a checkpoint is a snapshot of the model at a specific training step
    checkpoint_dir ="mancoll_bert2/checkpoint-845" # loads a saved trained model from defined folder, to evaluate the saved best model
    # load a saved fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, 
        num_labels=num_labels, 
        id2label=id2label,
        label2id=label2id,
    )

    # create an object based on the Trainer class that runs forward passes and gets predictions. We use Trainer instead of Weighted trainer as we dont calculate loss in prediction
    tmp_trainer = Trainer(
        model=model, # load the model from above
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    time_start = time.time() # store current time to calulate running time
    preds = tmp_trainer.predict(dataset) # runs the model on the input dataset and gives predictions, preds contains predictions (logits), true labes and evaluation results/metrics
    y_true = preds.label_ids # extract true labels from prediction output
    y_pred = np.argmax(preds.predictions, axis=1) # extract predictions, logits which are raw scores from the model for each class, the logit with the highest score is chosen, and that index represents the predicted class. Ex. logits for one sample if we have three classes: [2.1, 0.3, -1.2], then index 0 is highest, and we predict class 0
    records = [] # create an empty list, can be moved further down

    # print accuracy and macro F1 score
    print(f"\n== {name} ==") 
    print(f"Accuracy (all): {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro (all): {f1_score(y_true, y_pred, average='macro'):.4f}")

    # dictionary that translates numbers into class names
    id2label2 = {
        0: "Rear-End",
        1: "Head-On",
        2: "Side-Swipe",
        3: "Angle",
        4: "Other",
        5: "Parked",
        6: "Unknown"
    }

    # prints classification report including precision, recall, F1-score and score for each collision type using the true and predicted labels
    print(classification_report(
        y_true,
        y_pred,
        target_names=[id2label2[i] for i in range(num_labels)]
    ))

    # creates a list of dictionaries, one for each sample, containing summary (which is * for now), true label and prediction
    for i in range(3500): # THIS LINE SHOULD BE CHANGED TO: for i in range(len(y_true)): to ensure it matches the number of samples we have
        records.append({
            'SUMMARY': "*",
            'MANCOLL': y_true[i],
            'collision_type': y_pred[i]
        })

    result_df = pd.DataFrame(records) # converts the list of dictionaries to a dataframe
    output_path = "mancoll_bert2/bert_test_results-845-4.xlsx" # define where the file should be saved and the name
    dir_path = os.path.dirname(output_path) # extracts the folder name from the full path
    if dir_path and not os.path.exists(dir_path): # If the folder does not exist, then create it
        os.makedirs(dir_path)

    result_df.to_excel(output_path, index=False) # saves the DataFrame as an Excel file in output_path
    # 去掉 Unknown
    # removes all samples where the true label is class 6, “Unknown”, to evaluate the model without that class
    mask = y_true != 6
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    time_end = time.time() # store current time to calculate running time 

    # print accuracy, f1 macro and running time without class 6
    print(f"\n-- Excluding 'Unknown' class --")
    print(f"Accuracy (no Unknown): {accuracy_score(y_true_filtered, y_pred_filtered):.4f}")
    print(f"F1-macro (no Unknown): {f1_score(y_true_filtered, y_pred_filtered, average='macro'):.4f}")
    print("***")
    print(time_end - time_start)

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


# if __name__ == "__main__":
#     examples = [
#         "V1 struck the rear of V2 pushing V2 into V3. V3 then left the scene.",
#         "Two vehicles collided head-on on a two-lane road.",
#     ]
#     print("\nPredictions:", predict_texts(examples))
