# -*- coding: utf-8 -*-
"""
Custom training script for running each SageMaker training job in the experiment. 
Using the HuggingFace Trainer() API, and a pretrained HuggingFace AutoModel.
"""
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch
import pandas as pd
import numpy as np
import gc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_labels", type=int, default = 2)
    parser.add_argument("--learning_rate", type=str, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    # additional parameters
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    val_dataset = load_from_disk(args.val_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded val_dataset length is: {len(val_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    
    def compute_metrics(raw_preds): 
        ''' Generates metrics for classification for use in training and evaluation by the Trainer.
        
        Parameters:
        ----------
        raw_preds: NamedTuple
             var assigned in training to Trainer.predict() containing keys for: predictions, label_ids, metrics
                   
        Returns:
        --------
        metrics: dictionary containing...
            accuracy: float
                average accuracy across classes
            
            f1_score: float
                f1 score, macro weighted
            
            precision: float
                precision, macro weighted
            
            recall: float
                recall, macro weighted
            
            cls_rep: string / dict
                text summary of the classification metrics 

            conf_mat: ndarray of shape (n_classes, n_classes)
                confusion matrix, typed as str for json serialization 
        
        '''
        labels = raw_preds.label_ids 
        smooth_preds = raw_preds.predictions.argmax(-1) 
        precision, recall, f1, _ = precision_recall_fscore_support(labels, smooth_preds, average='macro')
        acc = accuracy_score(labels, smooth_preds)
        conf_mat = confusion_matrix(labels, smooth_preds)
        cls_rep = classification_report(labels, smooth_preds)
        
        return {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall, "confusion_matrix":str(conf_mat), "classification_report":cls_rep}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    

    
    # train model
    trainer.train()
    
    # define functions for evaluating model results and writing to file
    
    def eval_results(trainer, dataset):
        ''' Runs an evaluation loop and returns metrics with full float values.
        
        Parameters:
        -----------
        trainer: instantiation of HuggingFace Trainer object
            has been trained on desired training set
            
        dataset: Datasets.dataset object
            contains split of the data to evaluate
        
        Returns:
        --------
        trainer.evaluate(eval_dataset = dataset): dict
            str:float key-value pairs for evaluation loss and other metrics computed from compute_metrics()
        '''
        return trainer.evaluate(eval_dataset = dataset)
    
              
    def write_results(split, trainer, dataset):
        ''' Write the results of the evaluation to file in S3 at predefined output_data_dir.
        
        Parameters:
        ----------
        split: string
            describe which data split is being passed to add to file names
        
        trainer: instantiation of HuggingFace Trainer object
            has been trained on desired training set
            
        dataset: datasets.Dataset
            contains data split to write results on
        
        Returns:
        --------
        None: writes results of the model predictions to file
        
        '''
        
        eval_result = eval_results(trainer, dataset)
        
        # writes to file which can be accessed later in s3 ouput 
        with open(os.path.join(args.output_data_dir, f"{split}_eval_results.txt"), "w") as writer:

            writer.write(f"***** {split} eval results *****")
    
            for key, value in sorted(eval_result.items()):

                # adjust formatting where needed for readability
                if key in ["eval_confusion_matrix", "eval_classification_report"]:
                    writer.write(f"\n{split}_{key}:\n{value}\n")
                
                else:
                    writer.write(f"\n{split}_{key} = {value}\n")
        
        return 
    
    # write ultimate result reports to output_data_dir in s3
    write_results("val", trainer, val_dataset)
    write_results("test", trainer, test_dataset)              
    

    # Saves the model to s3
    trainer.save_model(args.model_dir)

    