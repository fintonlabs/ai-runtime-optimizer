from setuptools import setup, find_packages

setup(
    name='performance-analyzer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.2',
        'pandas==1.3.3',
        'scikit-learn==0.24.2',
        'tensorflow==2.6.0'
    ],
)