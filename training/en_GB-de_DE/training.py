import random
import datasets

import numpy as np
import pandas as pd

from datetime import datetime
from IPython.display import display, HTML
from data_preprocessing import process_data
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianTokenizer



def load_data(): # from BQ/bucket?
    """
    Output: processed dataset for model training
    :param dataset: raw dataset to transform
    :param source_len: maximum sentence length for source string
    :param target_len: maximum sentence length for target string
    :param source: source language
    :param target: target language
    """
    dataset = process_data()
    
    return dataset


def tokenization_processing(dataset, source_len=128, target_len=128,
                            source="en_GB", target="de_DE"):
    """
    Output: Generates tokenized data using the attributes of the base model
    :param dataset: raw dataset to transform
    :param source_len: maximum sentence length for source string
    :param target_len: maximum sentence length for target string
    :param source: source language
    :param target: target language
    """
    
    inputs = [s[source] for s in dataset["translation"]]
    targets = [s[target] for s in dataset["translation"]]
    
    model_inputs = tokenizer(inputs, max_length=source_len, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=target_len, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def compute_metrics(predictions):
    """
    Output: evaluation metrics to track model performance in training
    :param predictions: output of predictions to decode
    """
    
    metric = load_metric("sacrebleu")
    
    def process_text(predictions, labels):
        preds = [pred.strip() for pred in predictions]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    preds, labels = predictions
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = process_text(decoded_preds, decoded_labels)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    
    result = {k: round(v, 6) for k, v in result.items()} #round results
    
    return result


def pipeline(model_name="../../models/en_GB-de_DE/IKEA-MT_en-GB_de-DE_2022-08-01 08:36:40.937862/",
             model_output_name='IKEA_MT', batch_size=16, learning_rate=2e-5, 
             weight_decay=0.01, save_limit=20, epochs=5):
    """
    Output
    :param:
    """
    global tokenizer, trained_model
    
    print('load data')
    dataset = load_data()
    
    print('load models and data collator')
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=trained_model)

    
    print('map model inputs')
    tokenized_dataset = dataset.map(tokenization_processing, batched=True)
    
    time = str(datetime.now())
    
    args = Seq2SeqTrainingArguments(
        f"../../models/checkpoints/{model_output_name}_en-de_{time}",
        evaluation_strategy = "epoch",
        learning_rate = learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay = weight_decay,
        save_total_limit = save_limit,
        num_train_epochs = epochs,
        predict_with_generate=True    
    )
    
    trainer = Seq2SeqTrainer(
        trained_model,
        args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["validation"],
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    print('train model')
    trainer.train()

    print('save model')
    trainer.save_model('../../models/en_GB/IKEA_MT-en_GB-de_DE_' + time)
    
    return trainer