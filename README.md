# CVAT DATA FLOW

## Overview

This utility facilitates the download of datasets from [CVAT](https://github.com/opencv/cvat), preparing them for training and testing purposes. Key features include:
- Downloading datasets from CVAT in any supported format.
- Merging multiple tasks to form a consolidated dataset.
- Randomly segmenting the dataset into training, validation, and testing subsets.

## Local Installation

Clone the repository:

```bash
git clone https://github.com/ai-iaguzhinskii/cvat_data_flow.git
```

Install the dependencies:

```bash
pip3 install -r requirements.txt
```

## Configuration

Edit the configuration as per your requirements in the [config_file](config.ini):

```ini
[CVAT]
URL = # URL of the CVAT server, e.g., http://localhost:8080
LOGIN = # CVAT user login, e.g., admin
PASS = # CVAT user password, e.g., admin

[DOWNLOAD]
TASKS_IDS = # Task IDs for downloading, e.g., [111, 222, 333]
PROJECTS_IDS = # Project IDs for downloading, e.g., [111, 222, 333]

[DATASET]
FORMAT = # Desired dataset format. Refer to CVAT documentation for options, e.g., coco
SAVE_PATH = # Directory to save the downloaded datasets, e.g., /home/user/datasets
SPLIT = # Define random splits for the dataset, e.g., {"train": 0.8, "val": 0.1, "test": 0.1}

[OPTIONS]
ONLY_BUILD_DATASET = # Toggle to build dataset only if tasks are previously downloaded, e.g., True
LABELS_MAPPING = # Define label mappings, e.g., {"car": "vehicle", "person": "pedestrian"}
DEBUG = # Toggle debug mode, e.g., True
```

## Usage

To run the program:

```bash
python3 cvat_data_flow/main.py
```

Note: Ensure to replace placeholders in the configuration with actual values before executing.

## Installation via PIP

The utility can be installed via pip:

```bash
pip3 install cvat-data-flow
```

## Usage as a Python Package

The utility can also be used as a Python package. Here's an example:

```python
from cvat_data_flow import CVATDataFlow

cvat_data_flow = CVATDataFlow(
        url='http://cvat.example.com',
        login='username',
        password='password',
        save_path='/path/to/save/dataset',
        projects_ids=[1, 2, 3],
        only_build_dataset=False,
        format='coco',
        split=[('train', 0.7), ('val', 0.2), ('test', 0.1)],
        labels_mapping={'person': 'person', 'car': 'vehicle'}
    )

cvat_data_flow.download_data()
cvat_data_flow.build_dataset()
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.