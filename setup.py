# -*- coding: utf-8 -*-
"""
Make modules into a package. Default from the Cookie Cutter Data Science package.
"""
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This is a project to demonstrate distributed training and fine tuning of HuggingFace models with Sagemaker',
    author='Samantha Stuart, Professional Services I Intern, Amazon Web Services',
    license='MIT',
)
