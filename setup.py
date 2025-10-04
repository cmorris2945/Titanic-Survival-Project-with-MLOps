from setuptools import setup, find_packages

setup(
    name="titanic-mlops-project",
    version="0.1.0",
    description="Titanic MLOps Pipeline with Airflow",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "setuptools",
        "apache-airflow-providers-google==10.10.1",
        "sqlalchemy",
        "psycopg2-binary",
        "apache-airflow-providers-postgres>=5.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.8",
)