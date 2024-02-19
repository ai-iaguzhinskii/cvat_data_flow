from setuptools import setup, find_packages

setup(
    name="cvat_data_flow",
    version="0.0.1",
    author="Aleksei Iaguzhinskii",
    description="A utility for working with CVAT data flow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ai-iaguzhinskii/cvat_data_flow",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
    install_requires=open("requirements.txt").read().splitlines(),
)
