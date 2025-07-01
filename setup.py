from setuptools import setup, find_packages

setup(
    name="qualitylab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn==1.6.1",
        "streamlit",
        "seaborn",
        "matplotlib",
        "joblib",
        "openpyxl",
        "xlrd",
        "lime",
        "upsetplot",
        "click",
    ],
    entry_points={
        "console_scripts": [
            "qualitylab=qualitylab.cli:cli",
        ]
    }
)
