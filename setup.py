from setuptools import setup, find_packages

setup(
    name="qualitylab",
    version="0.1.0",
    py_modules=[
        "build_time",
        "build_quantity",
        "defects",
        "feature_engineering",
        "cli",
        "spreadsheets",
        "streamlit_app",
    ],
    install_requires=[
        "pandas",
        "scikit-learn",
        "streamlit",
        "seaborn",
        "matplotlib",
        "joblib",
        "openpyxl",
        "xlrd"
    ],
    entry_points={
        "console_scripts": [
            "qualitylab=cli:cli",
        ]
    }
)
