# -*- coding: utf-8 -*-
"""
Contains functions that can be called to execute a SageMaker training job using the Python SDK, based on the experimental design passed to the function.
"""
# import python libraries
import pandas as pd
from time import gmtime, strftime 
import json
import os

# import aws/sagemaker resources
import boto3
import sagemaker
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace
# from sagemaker import get_execution_role # uncomment if in SageMaker Notebooks


# import huggingface resources
import datasets
from datasets import load_dataset
from datasets.filesystems import S3FileSystem
from transformers import AutoTokenizer

def run_experiment(exp_design_path = '../data/interim/test_4_node_experimental_design.csv'):
    '''Prepares and executes the experiment specified at the experimental design path. All configurations are made
    based on the experimental design passed to the function.
    
    Parameters:
    ----------
    exp_design_path: string
        relative path to the csv containing the experimental design with hyperparameters to pass to the training job
        
    Returns:
    -------
    None:
        executes a SageMaker training job defined in src/models/train_model.py to collect data (the "experiment")
    '''

    # Read experimental design
    exp_design = pd.read_csv(exp_design_path)

    # Configure AWS Resources

    # S3 bucket and connection
    s3 = S3FileSystem() 
    bucket = exp_design['s3_bucket'].unique()[0] # should only have one configured

    # dataset used from HuggingFace Website
    dataset_name = exp_design['dataset_name'].unique()[0] # should only have one configured

    # set tokenizer used in preprocessing
    tokenizer_name = exp_design['automodel_name'].unique()[0] # should only have one configured
    checkpoint = tokenizer_name # ensures checkpoint and tokenizer are the same, required for an AutoModel

    # Define IAM role
    iam_client = boto3.client('iam')
    role_name = os.getenv("AWS_ROLE")
    role = iam_client.get_role(RoleName=role_name)['Role']['Arn']
    sess = sagemaker.Session(default_bucket=f"{bucket}")
    metrics_session = sagemaker.session.Session() # use to get metrics after training
    # if in SageMaker Notebooks, comment out role_name and role above and replace with below:
    # role = get_execution_role()

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
    parallelism_bool_run_order = exp_design['parallel_enabled']
    instance_type_run_order = exp_design['instance_type']
    epochs_run_order = exp_design['epochs']
    batch_run_order = exp_design['per_device_train_batch_size']
    lr_run_order = exp_design['learning_rate']
    volume_size_run_order = exp_design['volume_size']
    num_runs = len(node_run_order)
    run_range = exp_design['run_id']

    print("Settings for this experiment:")
    print(display(exp_design))

    # set dataset sizes of interest
    dataset_sizes = size_run_order.unique() # includes training and val obs to be loaded

    print("Dataset sizes:", dataset_sizes)

    # define metric to be tracked for a response, scrapes CloudWatch log files according to pattern
    metric_definitions = [{'Name': 'f1_score', 'Regex': '\'eval_f1_score\': ([0-9\\.]+)'}]

    # execute experiments in order of design passed
    for run_id in range(0, num_runs):

        run_number = run_range.iloc[run_id] # defined in experimental design
        print("Preparing to initiate experiment w/ run number: {}".format(run_number))

        # set hyperparameters, which are passed into the training job
        hyperparameters = {'epochs': int(epochs_run_order.iloc[run_id]),
                         'train_batch_size': int(batch_run_order.iloc[run_id]),
                         'learning_rate': float(lr_run_order.iloc[run_id]),
                         'model_name':checkpoint,
                         }
        run_job_name = "experiment-run{}-{}".format(run_number, strftime("%d-%H-%M-%S", gmtime()))

        if parallelism_bool_run_order.iloc[run_id] == False: # no parallelism desired

            huggingface_estimator = HuggingFace(entry_point='train_model.py',
                                source_dir='../src/models/',
                                output_path = output_path,
                                code_location = output_path,
                                instance_type=instance_type_run_order.iloc[run_id],
                                instance_count=int(node_run_order.iloc[run_id]),
                                volume_size = int(volume_size_run_order.iloc[run_id]),
                                role=role,
                                image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04',
                                py_version = 'py36',
                                metric_definitions = metric_definitions,
                                hyperparameters = hyperparameters)     

        else: # configuration for data parallelism
            dp_distribution = {'smdistributed':{'dataparallel':{ 'enabled': True }}}

            huggingface_estimator = HuggingFace(entry_point='train_model.py',
                                source_dir='../src/models/',
                                output_path = output_path,
                                code_location = output_path,
                                instance_type=instance_type_run_order.iloc[run_id],
                                instance_count=int(node_run_order.iloc[run_id]),
                                volume_size = int(volume_size_run_order.iloc[run_id]),
                                role=role,
                                image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04',
                                py_version = 'py36',
                                metric_definitions = metric_definitions,
                                hyperparameters = hyperparameters,
                                distribution = dp_distribution)  

        print("Starting Training Job.")
        training_input_path = f's3://{bucket}/datasets/{dataset_name}/{int(size_run_order.iloc[run_id])}/train'
        validation_input_path = f's3://{bucket}/datasets/{dataset_name}/{int(size_run_order.iloc[run_id])}/val'
        test_input_path = f's3://{bucket}/datasets/{dataset_name}/test'

        print("Train input path:", training_input_path)

        huggingface_estimator.fit({'train': training_input_path, 'val': validation_input_path, 'test': test_input_path}, job_name = run_job_name)

        fetch_train_job_results(run_job_name, run_number)
        
    return
    
def fetch_train_job_results(run_job_name, run_number):
    '''After training job execution, fetch results for recording in the experiment.
    
    Parameters:
    -----------
    run_job_name: string
    
    run_number: int
    
    Returns:
    -------
    None: 
        writes JSON results (run#_results.txt) for the given experiment 
        to file in data/interim directory

    '''

    metrics_session = sagemaker.session.Session() # use to get metrics after training

    print("Training job finished. Fetching metrics.")
    train_metrics = sagemaker.TrainingJobAnalytics(run_job_name).dataframe()
    run_f1 = train_metrics['value'].values[0]
    print("Run {} \nF1: {}".format(run_number, run_f1))

    # get train time and billable seconds
    job_description = metrics_session.describe_training_job(run_job_name)
    run_train_time = job_description['TrainingTimeInSeconds']
    run_bill_time = job_description['BillableTimeInSeconds']

    print("Train Time:", run_train_time, "Bill Time:", run_bill_time)

    # write results to file
    run_results = {"run_number":int(run_number), "job_name":run_job_name, "training_time":run_train_time, "bill_time":run_bill_time, "mean_f1":float(run_f1)}

    with open('../data/interim/run{}_results.txt'.format(run_number), 'w') as convert_file:
        convert_file.write(json.dumps(run_results))

    print("Experiment {} complete. Results written to ..data/interim folder.".format(run_number))
    
    return 


