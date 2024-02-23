# Description: Class for merging and transforming datasets in datumaro format.
import os
from collections import OrderedDict
import json
import logging
from datumaro import Dataset, HLOps, AnnotationType
from datumaro.plugins.data_formats.coco.exporter import _InstancesExporter, CocoTask, cast
from .utils.coco_converter import COCOConverter

# patch coco _InstancesExporter to save categories in the same order as in the dataset
class PatchInstancesExporter(_InstancesExporter):
    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return

        label_categories_name_ind = dataset.categories().get(AnnotationType.label)._indices
        for _, cat in enumerate(label_categories.items):
            self.categories.append(
                {
                    "id": 1 + label_categories_name_ind[cat.name],
                    "name": cast(cat.name, str, ""),
                    "supercategory": cast(cat.parent, str, ""),
                }
            )

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
        self.method_split = 'random_split'
        self.method_mapping = 'remap_labels'

        self.splits = splits
        self.mapping = mapping
        self.labels_id_mapping = labels_id_mapping

    
    def _create_dataset(self) -> Dataset:
        """
        Create datumaro dataset from tasks in datumaro format
        """
        source_datasets = self._create_source_datasets()
        merged_dataset = HLOps.merge(*source_datasets)
        return merged_dataset

    def _create_source_datasets(self) -> list:
        """
        Create list of Dataset from tasks in datumaro format
        """
        source_datasets = []
        for name in self.datasets_names:
            source_datasets.append(Dataset.import_from(os.path.join(self.datasets_path, name), 'datumaro'))
        return source_datasets
    
    def _mapping_labels(self, dataset: Dataset, mapping: list) -> Dataset:
        """
        Mapping labels from source to target
        
        :param dataset: datumaro.dataset
        :param mapping: list of tuple of mapping discription: [('source_label': 'target_label')].

        :return filter_dataset: ProjectDataset
        """
        
        mapped_dataset = dataset.transform(self.method_mapping, mapping=mapping)

        return mapped_dataset
    
    def _export_coco(self, dataset: Dataset, path: str):
        """
        Export dataset in COCO format

        :param dataset: Dataset in datumaro format
        :param path: path to save dataset
        """

        # patch exporter to save categories in the same order as in the dataset
        exporter = dataset.env.exporters['coco']
        exporter._TASK_CONVERTER[CocoTask.instances] = PatchInstancesExporter

        # save in COCO format
        dataset.export(save_dir=path, format=exporter, save_media=True)
    
    def _export_yolo(self, dataset: Dataset, path: str):
        """
        Export dataset in YOLO format

        :param dataset: Dataset in datumaro format
        :param path: path to save dataset
        """

        # save in COCO format
        self._export_coco(dataset, path)

        use_segments = False
        if "seg" in self.export_format:
            use_segments = True
        
        # convert to YOLO format
        coco2yolo = COCOConverter(
            json_dir=os.path.join(path, 'annotations'),
            save_dir=os.path.join(path, 'labels'),
            use_segments=use_segments,
            convert_format='yolo',
            labels_id_mapping=self.labels_id_mapping
        )

        coco2yolo.convert()

        # remove COCO format
        os.system(f'rm -rf {path}/annotations')
    
    def _reindex_labels(self, dataset: Dataset, labels_id_mapping: dict) -> Dataset:
        """
        Reindex labels.

        :param dataset: Dataset in datumaro format
        :param labels_id_mapping: dict of mapping discription: {'label_name': target_id}

        :return reindexed_dataset: ProjectDataset
        """

        category_dict_name_ind = dataset.categories().get(AnnotationType.label)._indices
        category_dict_ind_name = {v: k for k, v in category_dict_name_ind.items()}

        for item in dataset:
            for annotation in item.annotations:
                annotation.label = labels_id_mapping[category_dict_ind_name[annotation.label]]
                
        dataset.categories().get(AnnotationType.label)._indices = labels_id_mapping

        return dataset
    
    def export_dataset(self):
        """
        Transform and save dataset in specified format
        """

        if self.splits is not None:
            self.dataset = self.dataset.transform('random_split', splits=self.splits)
        
        if self.mapping is not None:
            self.dataset = self.dataset.transform(self.method_mapping, mapping=self.mapping)

        if self.labels_id_mapping is not None:
            self.dataset = self._reindex_labels(self.dataset, self.labels_id_mapping)

        if 'yolo' in self.export_format:
            self._export_yolo(self.dataset, f'{self.datasets_path}_{self.export_format}')
        elif 'coco' in self.export_format:
            self._export_coco(self.dataset, f'{self.datasets_path}_{self.export_format}')
        else:
            self.dataset.export(save_dir=f'{self.datasets_path}_{self.export_format}', format=self.export_format, save_media=True)
        
            