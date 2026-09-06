import os
from datasets import Dataset, load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration
import numpy as np
import torch

torch.cuda.empty_cache()

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Définir l'appareil (utilisation de CUDA si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialiser le DatasetDict
common_voice = DatasetDict()

# Charger le dataset
common_voice["train"] = load_dataset("Hani89/medical_asr_recording_dataset", split="train",
                                     token=os.environ.get("HF_TOKEN"))
common_voice["test"] = load_dataset("Hani89/medical_asr_recording_dataset", split="test",
                                    token=os.environ.get("HF_TOKEN"))


# Fonction pour diviser un dataset en n parties
def split_dataset(dataset, n_splits=5):
    splits = []
    split_size = 1 / n_splits
    for i in range(n_splits):
        if i == n_splits - 1:  # S'assurer que la dernière division obtient toutes les données restantes
            split = dataset.train_test_split(test_size=split_size, shuffle=True)
        else:
            split = dataset.train_test_split(test_size=split_size, shuffle=True)
            dataset = split['train']
        splits.append(split['test'])
    return splits


# Diviser les datasets de train et test en 20 parties chacun
train_splits = split_dataset(common_voice["train"], n_splits=20)
test_splits = split_dataset(common_voice["test"], n_splits=20)

# Utiliser seulement la 20ème partie des datasets train et test
train_split_20 = train_splits[19]
test_split_20 = test_splits[19]

# 2. Initialiser le tokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

# 4. Initialiser le processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")


# 7. Fonction pour préparer le dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# 8. Initialiser le feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# Préparer les datasets
train_split_20 = train_split_20.map(prepare_dataset, remove_columns=train_split_20.column_names, num_proc=1)
test_split_20 = test_split_20.map(prepare_dataset, remove_columns=test_split_20.column_names, num_proc=1)

# 10. Initialiser le modèle et le déplacer vers le GPU
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)


# 12. Définir le DataCollator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt").to(device)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt").to(device)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# 14. Charger le métrique WER
metric = evaluate.load("wer")

# 16. Fonction pour calculer les métriques
"""def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
"""

"""def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids

    # Debug information
    print(f"Pred_ids type: {type(pred_ids)}, shape: {pred_ids.shape if isinstance(pred_ids, np.ndarray) else 'N/A'}")
    print(
        f"Label_ids type: {type(label_ids)}, shape: {label_ids.shape if isinstance(label_ids, np.ndarray) else 'N/A'}")

    # Replace -100 with the pad_token_id in label_ids
    if isinstance(label_ids, np.ndarray):
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
    else:
        label_ids = [np.where(np.array(l) == -100, tokenizer.pad_token_id, l).tolist() for l in label_ids]

    # Ensure pred_ids and label_ids are in the correct shape
    if len(pred_ids.shape) > 2:
        pred_ids = pred_ids.reshape((-1, pred_ids.shape[-1]))
    if len(label_ids.shape) > 2:
        label_ids = label_ids.reshape((-1, label_ids.shape[-1]))

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
"""


def compute_metrics(pred):
    pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# 17. Définir les arguments de l'entraînement
from transformers import Seq2SeqTrainingArguments

# Ajustez les hyperparamètres comme suit :
"""training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Taille de batch plus petite
    gradient_accumulation_steps=2,
    learning_rate=5e-5,  # Learning rate ajusté
    warmup_steps=200,
    max_steps=200,  # Plus d'étapes pour permettre un meilleur entraînement
    gradient_checkpointing=False,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=25,
    save_steps=200,
    eval_steps=200,
    logging_steps=20,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_pin_memory=False
)"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    fp16=True,
    predict_with_generate=True,
    dataloader_pin_memory=False
)

# 18. Initialiser l'entraîneur
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split_20,
    eval_dataset=test_split_20,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 19. Lancer l'entraînement
trainer.train()
