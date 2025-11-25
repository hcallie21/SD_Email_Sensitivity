# OBJECTIVE: finetune/train TinyBERT, BERT, KNN, and SVM on synthetic JSON dataset

# Imports
from pyexpat import model
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
import sys 

# Load data
print("Loading data...")
df = pd.read_json("..\\sentences.jsonl", lines = True)  
texts = df["text"].tolist()
labels = df["sensitive"].tolist()

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Train/test split
print("Splitting data...")
X_train_texts, X_test_texts, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


print("Tokenizing data...")


def tokenize_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Tokenized datasets
train_encodings = tokenize_texts(X_train_texts)
test_encodings = tokenize_texts(X_test_texts)

# PyTorch dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)

# Fine-tune TinyBERT
tinybert_model = AutoModelForSequenceClassification.from_pretrained(
    "huawei-noah/TinyBERT_General_4L_312D", num_labels=len(le.classes_)
)
print("Fine-tuning TinyBERT...")


training_args = TrainingArguments(
    output_dir="./results",
   # evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True)

    # flatten the metrics
    flat = {}
    for label, stats in report.items():
        if isinstance(stats, dict):
            for k, v in stats.items():
                flat[f"{label}_{k}"] = v
        else:
            flat[label] = stats

    return flat

trainer = Trainer(
    model=tinybert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
# metrics = trainer.evaluate()
# print("TinyBERT Results:")
# print(classification_report(y_test, np.argmax(trainer.predict(test_dataset).predictions, axis=1)))
metrics = trainer.evaluate()
print("TinyBERT Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Pretty human-readable report
preds = np.argmax(trainer.predict(test_dataset).predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Extract embeddings for classical ML (KNN/SVM)
def get_embeddings(model, texts):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        outputs = model(**encodings)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Use base TinyBERT embeddings
base_model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
X_train_emb = get_embeddings(base_model, X_train_texts)
X_test_emb = get_embeddings(base_model, X_test_texts)


# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_emb, y_train)
y_pred_knn = knn.predict(X_test_emb)
print("KNN Results:")
print(classification_report(y_test, y_pred_knn))

# Train SVM
svm = SVC(kernel="linear")
svm.fit(X_train_emb, y_train)
y_pred_svm = svm.predict(X_test_emb)
print("SVM Results:")
print(classification_report(y_test, y_pred_svm))
