# -*- coding: utf-8 -*-
"""
Merge individual result files with the planned experimental design to create a single all-encompassing 
dataframe with experiment and results.
"""
import click
import logging
import pandas as pd
import glob
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn experimental data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    
        Parameters:
        ----------
        input_filepath: string
            location of experimental data

        output_filepath: string
            destination for processed results csv file

        Returns:
        -------
        None:
            writes results dataframe to csv in processed folder
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from generated experimental data')

    # initialize list to store results from experiment runs
    df_list = []
    result_paths = glob.glob(f"{input_filepath}/run*_results.txt")

    # read in dataframe results
    for path in result_paths:
        exp_result = pd.read_json(path, orient = 'index')
        exp_result = exp_result.transpose() # transpose to  be in row wise format
        df_list.append(exp_result)
    
    # concatenate into one df for analysis
    raw_results = pd.concat(df_list).sort_values('run_number')
    # rename columns to match exp design titles
    raw_results.columns = ["run_id","job_name", "train_time", "billable_seconds","f1"]
    raw_results = raw_results.reset_index(drop=True)

    # read in exp design to match experiments to the raw results
    exp_design = pd.read_csv(f'{input_filepath}/experimental_design.csv')

    # drop the empty columns for the results
    exp_design = exp_design.drop(labels=["train_time", "f1","billable_seconds","cost"], axis =1)
    exp_design = exp_design.reset_index(drop=True)

    # join raw results and experiments tables
    results = exp_design.merge(raw_results, on = "run_id", how = "left")

    # add job cost based on price to df
    job_cost = (results['billable_seconds']/3600)*(results['hourly_price'])*(results['num_nodes'])
    results["job_cost"] = job_cost

    # write dataframe to csv in processed folder
    results.to_csv(f"{output_filepath}/experiment_results.csv")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
