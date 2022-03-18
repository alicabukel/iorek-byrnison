from setuptools import find_packages
from setuptools import setup

setup(
    name="tc_trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    description="Telco churn model training application.",
)
