import random
import datasets

import numpy as np
import pandas as pd

from datetime import datetime
# from IPython.display import display, HTML
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianTokenizer

# https://huggingface.co/course/chapter7/4?fw=pt - reference for building tuning model


def load_data(): # from BQ/bucket?
    """
    Output: processed dataset for model training
    """
    dataset = process_data()
    
    return dataset


def process_data():
    """
    Output: Translated text of the source text
    :param src_text: source string
    """ 
    
    #Load data (needs to be changed to loading from BQ or bucket)
    df = pd.read_csv('../data/en_GB-it_IT.csv', engine='python')

    df = df[['source', 'target']] # will be redundent when setup 
    
    # Remove nulls/duplicates
    
    # df = language_rules(df)

    # Transform to HuggingFace Dataset
    data = Dataset.from_pandas(pd.DataFrame({'translation': df.to_dict('records')})) 

    # Split data into training sets
    train_test_valid = data.train_test_split(shuffle=True, seed=7, test_size=0.0015)
    test_valid = train_test_valid['test'].train_test_split(shuffle=True, seed=7, test_size=0.5)

    # Convert to train/validate/test
    dataset = DatasetDict({
        'train': train_test_valid['train'],
        'validation': test_valid['test'],
        'test': test_valid['train']})
     
    return dataset


def tokenization_processing(dataset, source_len=128, target_len=128,
                            source="en_GB", target="it_IT"):
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


def compute_metrics(predictions): # n.b. can add more metrics later
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


def pipeline(model_name="Helsinki-NLP/opus-mt-en-it",
             model_output_name='IKEA_MT_en_GB-it_IT', batch_size=16, learning_rate=2e-5, 
             weight_decay=0.01, save_limit=20, epochs=5):
    """
    Output fine-tuned model
    :param model_name: path to model
    :param model_output_name: model output name to follow naming convention
    :param batch_size: batch size for model input int
    :param learning_rate: optimisation learning rate float
    :param weight_decay: weight decay float
    :param save_limit: number of checkpoints to save (every 5000 steps)
    :param epochs: number of passes through the training data
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
    
    # n.b. need to add early stopping with longer training times
    args = Seq2SeqTrainingArguments(
        f"../models/checkpoints/{model_output_name}_en-de_{time}",
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
    trainer.save_model('../models/en_GB-it_IT/IKEA_MT-en_it_IT_' + time)
    
    return trainer