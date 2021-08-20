amazon-sagemaker-huggingface-benchmark 🤗
==============================

Project Motivation
------------
In 2021, a new partnership between Amazon Web Services and HuggingFace has resulted in the release of several cutting-edge Amazon SageMaker features, towards the aim of helping customers train and deploy natural language models quickly on AWS. This repository is focused on helping customers and data scientists dive deeper into the functionality of three of these new collaborative features for model training: HuggingFace Deep Learning Containers, using the HuggingFace + SageMaker Python SDK to build AutoModels, and making use of SageMaker Data Parallelism to further reduce training time.

Customers and data scientists alike are interested in adopting these new features to fine-tune pretrained HuggingFace models on their projects. However, anyone new to working with HuggingFace and Sagemaker shares the same common questions: 

* How much will using distributed data parallelism reduce my model training time?
* How much will the cost of running training jobs change?
* How much will model performance change, if at all, by using distributed data paralellism to reduce training time?
* How do all of these factors change again for different amounts of data used in the fine-tuning process?

The present repository is an experimentation hub to collect data that will answer these questions for customers. By setting up and running experiments that identify the model training time, performance, and training job cost, at different settings for number of GPUs used (vertical and horizonal scaling), and at different dataset sizes (number of samples). All experiments are run using a HuggingFace Deep Learning Container in SageMaker, with the Trainer() API and SageMaker Python SDK. 

Project Scope
------------
The scope of the experiment performed in this repo is as follows.

<b>Experiment Plan:</b>
<p align="left">
<img src="reports/figures/project_scope.png" width=60% height=60%>
</p>

<b>Experiment Runs in Detail:</b>
<p align="left">
<img src="reports/figures/experiment_runs_detailed.png" width=60% height=100%>
</p>

<b>Overview of Parameters Used:</b>
* HuggingFace Pretrained AutoModel: distilbert-base-uncased
* Dataset/Task: binary text classification, of sentiment polarity of Amazon product reviews
   * "amazon-polarity" dataset from HuggingFace Datasets Hub
* Dataset size: 
   * 100,000 samples 
   * 600,000 samples 
   * 1,000,000 samples
* Instance Types:
   * ml.p3.2xlarge (1 GPU)
   * ml.p3.8xlarge (4 GPU)
   * ml.p3.16xlarge (8 GPU - 1 Node using SageMaker Data Parallel)
   * ml.p3.16xlarge (16 GPU - 2 Nodes using SageMaker Data Parallel)
   * ml.p3.16xlarge (32 GPU - 4 Nodes using SageMaker Data Parallel)

* Experimental controls:
   * Per Device Batch Size: 32
   * Learning Rate: 5e-5 (unless otherwise specified)
   * Number of Steps (for a given global batch size, to allow comparing training jobs with different numbers of samples)

The above tables illustrate the default design. As the user of this repo, you are more than welcome to adjust the parameters based on your needs, as described in the workflow diagram below. 


How to Use This Repo
------------
<b>Set up environment:</b>
* `git clone` repo to desired directory using the Clone URL
* Open repository in desired IDE (tested with VSCode, SageMaker Notebooks)
   * Note: in SageMaker Notebooks, existing Conda PyTorch kernels meets majority of installation requirements already
   * To install any extra required packages, simply uncomment and run the provided cells at the start of each Jupyter notebooks containing `!pip install PACKAGE_NAME`
* If working in a local IDE such as VSCode, set up environment using makefile commands in Terminal:
   * `>>make environment`
   *  activate your virtual environment `>>source activate amazon-sagemaker-huggingface-benchmark`
   * If you are using VSCode, you may have to restart the IDE before you can select the newly created environment kernel for your Jupyter Notebook (shown below)
   <p align="center">
<img src="reports/figures/env_notebook_kernel.png" width=30% height=20%>
</p>

<b>1. To explore the results of the existing benchmark of distilBERT on SageMaker:</b>
* Navigate to notebooks/analyze_results.ipynb
* Explore data in the results table, and customize visualizations as desired

<b>2. To change the experimental design and run new jobs follow the below workflow:</b>
* Note: you will have to add your custom AWS configuration information in the .env file (role, s3 bucket)
<p align="left">
<img src="reports/figures/workflow.png" width=100% height=20%>
</p>

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands to expedite iteration, like `make experiment` or `make results`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Extra external information about the default experimental design.
    │   ├── interim        <- Individual experiment run files and results generated by training jobs
    │   ├── processed      <- The final, canonical results dataset for visualization and analysis.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── notebooks              <- Jupyter notebooks for running the experiments (getting results), and analyzing results (analyze_results). 
    │   └── run_experiment.py  <- Allows get_results notebook to run training jobs directly based on your experimental design
    │                                               
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Can contain in future any generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting or README
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to manipulate data (make experimental design, downloadfrom HuggingFace 
        │   │                   and upload to s3, generate results from train jobs)
        │   └── make_experiment.py
        │   └── make_results.py
        │   └── make_s3_upload.py
        │   └── wrangle_dataset.py 
        │
        ├── models         <- Scripts to train models via SageMaker training jobs.              
        │   └── train_model.py
        │
        └── visualization  <- Scripts for visualization in analyze_results.ipynb 
            └── visualize.py

Dataset Credit
------------
The dataset used in this experiment is the "Amazon Polarity" dataset from the HuggingFace Hub, used 
for binary text classification. The task predicts a positive or negative rating given text from an Amazon review.

See more about it here: https://huggingface.co/datasets/amazon_polarity 

Known Issues/Bugs
------------
During the design of this repo and initial data collection, some data was collected by cloning failed training jobs launched via the SDK as described in get_results Step 7, and adjusting their EBS volumes manually. This was neccessary to work around a bug that prevented custom EBS volume_size values from being passed into SageMaker. The runs executed in the console can be identified by the title of their associated training job (different than standard format in run_experiment.py). However running the jobs in the console vs the SDK does not impact the ultimate results of the training job. If you are designing a custom experiment, and notice your training jobs do not have the correct volume_size passed to them, consequently causing an ArchiveError and job failure, you can work around the issue by cloning the failed job and adjusting the EBS volume manually. The results of any training job you have executed in your account can be looked up manually by passing the name of the training job and run number while executing the last cell in the get_results notebook titled "Manual Results Lookup."


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>