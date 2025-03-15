import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="human-eval",
    py_modules=["human-eval"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(),
    package_data={'human_eval': ['data/HumanEval.jsonl.gz']},
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)
