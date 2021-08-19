# -*- coding: utf-8 -*-
"""
Command executable by the Makefile that reads the desired dataset from the HuggingFace Datasets Hub, 
tokenizes the data, splits it into train-val-test sets, and uploads the splits in an organized 
manner to S3. Prepares and loads the datasets prior to running an experiment. Same code as in wrangle_datasets.py
only callable from the Makefile.
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# import python resources
import pandas as pd

# import aws/sagemaker resources
import boto3
import sagemaker
import sagemaker.huggingface
import os

# import huggingface resources
import datasets
from datasets import load_dataset
from datasets.filesystems import S3FileSystem
from transformers import AutoTokenizer

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Reads in custom experimental design, loads data from the HuggingFace Hub, tokenizes data, train-val splits data, uploads splits in 
    an organized directory structure to S3 to be called in the training script. 
    
    Uploads the required datasets to S3 in a one-time action so that the same paths can be called in all your training jobs 
    automatically by reference to the dataset size in the experimental design.

    Parameters:
    ----------
    input_filepath: string
        path in the directory to locate experimental design to read
    
    Returns:
    --------
    None: 
        prints message indicating completion
    """
    logger = logging.getLogger(__name__)
    logger.info('uploading data to s3')

    # Read experimental design
    exp_design = pd.read_csv(f'{input_filepath}/experimental_design.csv')

    # Configure AWS Resources

    # S3 bucket and connection
    s3 = S3FileSystem() 
    bucket = exp_design['s3_bucket'].unique()[0] # should only have one bucket per experiment configured

    # dataset used from HuggingFace Website
    dataset_name = exp_design['dataset_name'].unique()[0] # should only have one dataset per experiment configured

    # Define IAM role
    iam_client = boto3.client('iam')
    role_name = os.getenv("AWS_ROLE")
    role = iam_client.get_role(RoleName=role_name)['Role']['Arn']
    sess = sagemaker.Session(default_bucket=f"{bucket}")

    print("Configuration settings:")
    print(f"sagemaker role arn: {role}")
    print(f"sagemaker bucket: {sess.default_bucket()}")
    print(f"sagemaker session region: {sess.boto_region_name}")

    # Configure File I/O for Experiment

    # s3 key prefix for the data
    input_prefix = f'datasets/{dataset_name}'
    output_prefix = f"output/{dataset_name}"

    # output path for train job results
    output_path = f's3://{sess.default_bucket()}/{output_prefix}'

    # capture order of operations for experiment
    size_run_order = exp_design['dataset_size']

    # set dataset sizes of interest
    dataset_sizes = size_run_order.unique() # includes training and val obs to be loaded

    print("Dataset sizes (i.e. number of samples):", dataset_sizes)

    # initialize dataset lookups
    train_dataset_lookup = dict()
    val_dataset_lookup = dict()

    # set tokenizer used in preprocessing
    tokenizer_name =  exp_design['automodel_name'].unique()[0] # should only have one automodel_name configured per experiment
    checkpoint = tokenizer_name # ensures checkpoint and tokenizer are the same, required for an AutoModel

    # download tokenizer
    print("Downloading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # change 'content' to the text column name for your dataset
    def tokenize(batch):
        return tokenizer(batch['content'], padding='max_length', truncation=True)
        
    # split and tokenize datasets, going to assign s3 paths to dictionary for lookup later
    print("Splitting and tokenizing data.")

    for num_obs in dataset_sizes:
        
        train_split = datasets.load_dataset(dataset_name, split=datasets.ReadInstruction(
        'train', from_=0, to=num_obs, unit='abs'))
        
        # split loaded training data into 90% training and 10% val sets, adjust test_size as desired
        train_val_dataset = train_split.train_test_split(test_size=0.1, seed = 324)
        
        # tokenize splits
        train_dataset = train_val_dataset['train'].map(tokenize, batched=True)
        val_dataset = train_val_dataset['test'].map(tokenize, batched=True)
            
        # set dataset format for pytorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        print("Uploading data to S3 for size: {}".format(num_obs))
        
        # save train_dataset to s3
        training_input_path = f's3://{sess.default_bucket()}/{input_prefix}/{num_obs}/train'
        train_dataset.save_to_disk(training_input_path,fs=s3)

        # save val_dataset to s3
        validation_input_path = f's3://{sess.default_bucket()}/{input_prefix}/{num_obs}/val'
        val_dataset.save_to_disk(validation_input_path,fs=s3)
        
        # assign s3 paths to tokenized splits to lookup dict for each dataset size
        train_dataset_lookup[num_obs] = training_input_path
        val_dataset_lookup[num_obs] = validation_input_path  
            
        print("Training/Val dataset paths for dataset size {}:".format(num_obs))
        print(f"{training_input_path}")
        print(f"{validation_input_path}")
        

    # load and tokenize test dataset
    # if your test data is not constant for all in the HF Hub, you may have to adjust this script
    test_dataset = load_dataset(dataset_name, split = 'test') # test split constant for all
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) # set format for pytorch

    # save test_dataset to s3
    test_input_path = f's3://{sess.default_bucket()}/{input_prefix}/test'
    print("Test dataset path:")
    print(f"{test_input_path}")
    test_dataset.save_to_disk(test_input_path,fs=s3)

    print("Upload completed! Check out your s3 bucket to see for yourself.")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
