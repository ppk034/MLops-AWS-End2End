from setuptools import setup, find_packages

setup(
    name="mlproject_end2end",             # Name of your project
    version="0.1.0",                      # Project version
    author="praveen",                   # Replace with your name
    description="End-to-end MLOps project",
    packages=find_packages(where="src"),  # Find packages in the src folder
    package_dir={"": "src"},              # Specify the src folder as the package directory
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "boto3",
        "flask",
        "gunicorn"
    ],
    python_requires=">=3.6",              # Specify the required Python version
)
