# Description: Class for merging and transforming datasets in datumaro format.
import os
from collections import OrderedDict
import json
import logging

import datumaro as dm
from datumaro.components.project import Project, ProjectDataset
from datumaro.components.operations import IntersectMerge
from datumaro.components.errors import QualityError, MergeError

from .utils.coco_converter import COCOConverter


class CustomDataset():
    """
    Class for merging and transforming datasets in datumaro format.

    Main methods:
    - transform_dataset

    Merge and transform datasets in datumaro format to the specified format and save it to the specified directory:
    Example:
    ```
    dataset = CustomDataset(
        datasets_path='/path/to/datasets',
        export_format='coco',
        splits=[('train', 0.7), ('val', 0.2), ('test', 0.1)],
        mapping=[('source_label', 'target_label')],
        labels_id_mapping={'label_name': target_id}
    )
    dataset.export_dataset()
    ```
    As a result, the dataset will be saved in '/path/to/datasets_coco_split' in the COCO format and will be divided into train, val and test subsets.
    """ 

    def __init__(self, 
                 datasets_path:str, export_format:str='coco', 
                 splits: list = None, mapping: tuple = None,
                 labels_id_mapping: dict = None):
        """
        :param datasets_path: path to folder with cvat tasks in datumaro format
        :param export_format: final format of dataset
        :param splits: list of tuple of spliting discription: ('subset_name': part_of_subset:float). Default: None
        :param mapping: list of tuple of mapping discription: ('source_label': 'target_label'). Default: None
        :param labels_id_mapping: dict of mapping discription: {'label_name': target_id}. Default: None
        """
        self.export_format = export_format
        self.datasets_path = datasets_path
        self.datasets_names = os.listdir(datasets_path)
        self.logger = logging.getLogger(__name__)

        self.dataset = self._create_dataset()
        self.method_split = self.dataset.env.make_transform('random_split')
        self.method_mapping = self.dataset.env.make_transform('remap_labels')

        self.splits = splits
        self.mapping = mapping
        self.labels_id_mapping = labels_id_mapping

    
    def _create_dataset(self) -> ProjectDataset:
        """
        Create datumaro dataset from tasks in datumaro format
        """
        source_datasets = self._create_projects()
        merged_dataset = self._merge_datasets(source_datasets)
        return merged_dataset

    def _create_projects(self) -> list:
        """
        Create list of ProjectDataset from tasks in datumaro format
        """
        source_datasets = []
        for name in self.datasets_names:
            project = Project().import_from(path=os.path.join(self.datasets_path, name),
                                            dataset_format='datumaro')
            source_datasets.append(project.make_dataset())
        return source_datasets
    
    def _merge_datasets(self, source_datasets:list=None, export:bool=False) -> ProjectDataset:
        """
        Build dataset from tasks without spliting

        :param source_datasets: list of ProjectDataset, see self._create_projects()

        :return merger_dataset: ProjectDataset
        """
        
        if source_datasets is None:
            source_datasets = self._create_projects()
        merger = IntersectMerge(conf=IntersectMerge.Conf(pairwise_dist=0.5, 
                                                         groups=[], 
                                                         output_conf_thresh=0.0, 
                                                         quorum=0)
        )
        merged_dataset = merger(source_datasets)

        return merged_dataset
    
    def _mapping_labels(self, dataset: ProjectDataset, mapping: list) -> ProjectDataset:
        """
        Mapping labels from source to target
        
        :param dataset: datumaro.dataset
        :param mapping: list of tuple of mapping discription: [('source_label': 'target_label')].

        :return filter_dataset: ProjectDataset
        """
        
        mapped_dataset = dataset.transform(self.method_mapping, mapping=mapping)
        mapped_dataset = mapped_dataset.filter(expr='/item/annotation', filter_annotations=True, remove_empty=True)

        return mapped_dataset
    
    def _export_yolo(self, dataset: ProjectDataset, path: str):
        """
        Export dataset in YOLO format

        :param dataset: ProjectDataset in datumaro format
        :param path: path to save dataset
        """

        # save in COCO format
        dataset.export(save_dir=path, format='coco', save_images=True)

        use_segments = False
        if "seg" in self.export_format:
            use_segments = True
        
        # convert to YOLO format
        coco2yolo = COCOConverter(
            json_dir=path,
            save_dir=path,
            use_segments=use_segments,
            convert_format='yolo'
        )

        coco2yolo.convert()

    def export_dataset(self):
        """
        Transform and save dataset in specified format
        """

        if self.splits is not None:
            self.dataset = self.dataset.transform(self.method_split, splits=self.splits)
        
        if self.mapping is not None:
            self.dataset = self._mapping_labels(self.dataset, self.mapping)

        if 'yolo' in self.export_format:
            self._export_yolo(self.dataset, f'{self.datasets_path}_{self.export_format}_split')
        else:
            self.dataset.export(f'{self.datasets_path}_{self.export_format}_split', self.export_format, save_images=True)
        
            