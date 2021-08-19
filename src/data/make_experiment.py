# -*- coding: utf-8 -*-
"""
Runs data processing scripts to turn raw experimental design plan csv from (../raw) into a
fully formulated experimental design with hyperparameters/customization to pass to a training job. 
"""
import click
import logging
import pandas as pd
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from csv import writer
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw experimental design csv from (../raw) into
        full formulated experimental design to pass to a training job in (saved in ../interim).
        
        INSTRUCTIONS:
        * Get started designing your experiment by following the steps explained in the get_results jupyter notebook. 
        * You can get started by exploring modifications to the default design in the Jupyter Notebook to suit your needs
        * Then, modify the code below to match your desired experimental design, for faster iteration 
        * To generate the neccessary files to start testing in Sagemaker (rather than running many Notebook cells) simply type "make experiment" in the terminal. 
        * To execute the experiments after you have run this, see the get_results Jupyter Notebook     
    
    Parameters:
    ----------
    input_filepath: string
        defined in MAKEFILE, location of input files
    output_filepath: string
        defined in MAKEFILE, location of output files

    Returns:
    -------
    None: generates files
        * experimental_design.csv (contains all your planned experiments + parameters)
        * individual csv files corresponding to each run (read into the training jobs individually as you run them)
    """
    logger = logging.getLogger(__name__)
    logger.info('making experimental design for training jobs from raw design')

    # load raw experimental design from user input
    exp_design = pd.read_csv(f'{input_filepath}/experimental_design.csv')

    # For the case of 1 GPU - num steps selected via guess and check to get whole numbered epochs
    num_steps = 84375 
    batch_size = 32*1 # for 1 GPU
    num_samples = exp_design['dataset_size'].values*0.9 # 90% of each size is used for training
    num_steps_per_epoch = num_samples/batch_size

    # calculate the number of epochs to be passed as hyperparameters for each experiment
    num_epochs = num_steps/num_steps_per_epoch
    # print("\nTraining dataset sample sizes:", num_samples)
    # print("\nRequired epochs for different sample sizes on 1 GPU:", num_epochs)
    s3_bucket = os.getenv("BUCKET_NAME")
    # insert additional columns containing info that the training jobs will require
    exp_design.insert(loc = 0, column = 'dataset_name', value = 'amazon_polarity')
    exp_design.insert(loc = 1, column = 'automodel_name', value = 'distilbert-base-uncased')
    exp_design.insert(loc = 2, column = 'num_parameters_tuned', value = 66955010) # constant for this model
    exp_design.insert(loc = 3, column = 's3_bucket', value = s3_bucket) # constant for this model
    exp_design.insert(loc = 4, column = 'per_device_train_batch_size', value = batch_size) # calculated above
    exp_design.insert(loc = 5, column = 'learning_rate', value = 5e-5)
    exp_design.insert(loc = 6, column = 'epochs', value = num_epochs) # calculated above

    # map the num_nodes column to specific factor levels for experimentation
    instance_mapper = {1:'ml.p3.2xlarge', 2:'ml.p3.16xlarge', 4:'ml.p3.16xlarge'}
    gpu_mapper = {1:1, 2:16, 4:32}
    parallel_enabled_mapper = {1:False, 2:True, 4:True}
    EBS_volume_mapper = {'ml.p3.2xlarge':1024, 'ml.p3.8xlarge':1024, 'ml.p3.16xlarge':30} # leave default for 16xlarges, add more storage for small instances
    price_mapper = {"ml.p3.2xlarge": 3.825, "ml.p3.8xlarge":14.688, "ml.p3.16xlarge":28.152} # hourly instance pricing from SageMaker website

    # keep adding extra desired info using the above mappers
    exp_design.insert(loc = 7, column = 'instance_type', value = exp_design['num_nodes'].map(instance_mapper))
    exp_design.insert(loc = 8, column = 'num_gpus', value = exp_design['num_nodes'].map(gpu_mapper))
    exp_design.insert(loc = 9, column = 'global_batch_size', value = exp_design['num_gpus']*exp_design['per_device_train_batch_size']) 
    exp_design.insert(loc = 10, column = 'num_steps', value = np.rint((exp_design['epochs'] * exp_design['dataset_size']*0.9)/exp_design['global_batch_size'])) # note - in the future num steps could be held constant across exps instead
    exp_design.insert(loc = 11, column = 'hourly_price', value = exp_design['instance_type'].map(price_mapper))
    exp_design.insert(loc = 12, column = 'volume_size', value = exp_design['instance_type'].map(EBS_volume_mapper)) 
    exp_design.insert(loc = 13, column = 'parallel_enabled', value = exp_design['num_nodes'].map(parallel_enabled_mapper)) 

    # save completed design to file
    exp_design.to_csv(f'{output_filepath}/experimental_design.csv', index_label = 'run_id')

    # OPTIONAL - add custom runs outside the initial design - comment out code section enclosed between lines -- if not desired --------------------
    # --------------------------------------------------------------------------------------------------------------------

    # if any additional data points desired, add additional rows to csv with custom params
    cp1 = [15,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,5.0,'ml.p3.8xlarge',4,128,21094,14.688,1024,False,1,600000]
    cp2 = [16,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',16,5e-05,5.0,'ml.p3.16xlarge',8,128,10547,28.152,1024,True,1,600000] # pd batch size adjusted for cuda error
    cp3 = [17,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,30.0,'ml.p3.16xlarge',8,256,10547,28.152,1024,True,1,100000]
    cp4 = [18,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,3.0,'ml.p3.16xlarge',8,256,10547,28.152,1024,True,1,1000000]
    cp5 = [19,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,30.0,'ml.p3.8xlarge',4,128,21094,14.688,1024,False,1,100000]
    cp6 = [20,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,3.0,'ml.p3.8xlarge',4,128,21094,14.688,1024,False,1,1000000]
    cp7 = [21,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,2.83e-4,30.0,'ml.p3.16xlarge',32,1024,2637,28.152,30,True,4,100000] # run tmo - sqrt law
    cp8 = [22,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,2.83e-4,3.0,'ml.p3.16xlarge',32,1024,2637,28.152,30,True,4,1000000] # run tmo - sqrt law
    cp9 = [23,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,30.0,'ml.p3.16xlarge',8,256,10547,28.152,1024,False,1,100000]
    cp10 = [24,'amazon_polarity','distilbert-base-uncased',66955010,'hf-benchmarking-samstu',32,5e-05,3.0,'ml.p3.16xlarge',8,256,10547,28.152,1024,False,1,1000000]

    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(f'{output_filepath}/experimental_design.csv', 'a') as f_object:
    
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(cp1)
        writer_object.writerow(cp2)
        writer_object.writerow(cp3)
        writer_object.writerow(cp4)
        writer_object.writerow(cp5)
        writer_object.writerow(cp6)

        # increased learn rate to compare with lr controlled version
        writer_object.writerow(cp7)
        writer_object.writerow(cp8)
        writer_object.writerow(cp9)
        writer_object.writerow(cp10)
  
        #Close the file object
        f_object.close()

    # read in updated file 
    exp_design = pd.read_csv(f'{output_filepath}/experimental_design.csv')
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    # export desired experiments into individual csvs
    for ix, val in exp_design.iterrows():
        exp_design.loc[exp_design.index == ix].to_csv(f'{output_filepath}/run{ix}_experimental_design.csv', index_label = 'run_id')
    
    # to execute experiments - follow instructions in the get_results notebook

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
