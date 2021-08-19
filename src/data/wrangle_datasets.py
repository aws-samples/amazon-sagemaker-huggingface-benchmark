# -*- coding: utf-8 -*-
"""
Same code that is referenced in make_s3_upload.py (see docstring there) only callable from Jupyter Notebook rather than the Makefile.
"""
# import python libraries
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

# Read experimental design
exp_design = pd.read_csv('../data/interim/experimental_design.csv')

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
metrics_session = sagemaker.session.Session() # use to get metrics after training

# assign region for the session
region = boto3.Session().region_name

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
node_run_order = exp_design['num_nodes']
epochs_run_order = exp_design['epochs']
batch_run_order = exp_design['per_device_train_batch_size']

num_runs = len(node_run_order)

print("Settings for this experiment:")
print(display(exp_design))

# set dataset sizes of interest
dataset_sizes = size_run_order.unique() # includes training and val obs to be loaded

print("Dataset sizes:", dataset_sizes)

# initialize dataset lookups
train_dataset_lookup = dict()
val_dataset_lookup = dict()

# set tokenizer used in preprocessing
tokenizer_name =  exp_design['automodel_name'].unique()[0] # should only have one automodel_name configured per experiment
checkpoint = tokenizer_name # ensures checkpoint and tokenizer are the same, required for an AutoModel

# download tokenizer
print("Downloading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def tokenize(batch):
    return tokenizer(batch['content'], padding='max_length', truncation=True)
    
# split and tokenize datasets, going to assign s3 paths to dictionary for lookup later
print("Splitting and tokenizing data.")

for num_obs in dataset_sizes:
    
    train_split = datasets.load_dataset(dataset_name, split=datasets.ReadInstruction(
    'train', from_=0, to=num_obs, unit='abs'))
    
    # split loaded training data into 90% training and 10% val sets
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
    

# load and tokenize test dataset, constant for all
test_dataset = load_dataset(dataset_name, split = 'test') # test split constant for all
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) # set format for pytorch

# save test_dataset to s3
test_input_path = f's3://{sess.default_bucket()}/{input_prefix}/test'
print("Test dataset path:")
print(f"{test_input_path}")
test_dataset.save_to_disk(test_input_path,fs=s3)
