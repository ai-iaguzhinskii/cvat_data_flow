# Description: Class for merging and transforming datasets in Datumaro format.

import os
import logging
from collections import OrderedDict
from datumaro import Dataset, HLOps, AnnotationType
from datumaro.plugins.data_formats.coco.exporter import _InstancesExporter, CocoTask, cast
from .utils.coco_converter import COCOConverter

# Patch Coco _InstancesExporter to save categories in the same order as in the dataset
class PatchInstancesExporter(_InstancesExporter):
    def save_categories(self, dataset):
        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            return

        label_categories_name_ind = dataset.categories().get(AnnotationType.label)._indices

        for name, ind in label_categories_name_ind.items():
            self.categories.append(
                {
                    "id": 1 + ind,
                    "name": cast(name, str, ""),
                    "supercategory": "",
                }
            )

class CustomDataset:
    """
    Class for merging and transforming datasets in Datumaro format.

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
    As a result, the dataset will be saved in '/path/to/datasets_coco_split' in the COCO format and will be divided into train, val, and test subsets.
    """

    def __init__(self, 
                 datasets_path: str, 
                 export_format: str = 'coco', 
                 splits: list = None, 
                 mapping: list = None, 
                 labels_id_mapping: dict = None):
        """
        Initialize the CustomDataset object.

        :param datasets_path: Path to folder with CVAT tasks in Datumaro format.
        :param export_format: Final format of the dataset.
        :param splits: List of tuples describing the data splits. Example: [('subset_name', part_of_subset: float)].
        :param mapping: List of tuples describing label mapping. Example: [('source_label', 'target_label')].
        :param labels_id_mapping: Dict mapping label names to target IDs. Example: {'label_name': target_id}.
        """
        self.datasets_path = datasets_path
        self.export_format = export_format
        self.splits = splits
        self.mapping = mapping
        self.labels_id_mapping = labels_id_mapping
        self.logger = logging.getLogger(__name__)

        self.dataset = self._create_dataset()

    def _create_dataset(self) -> Dataset:
        """
        Create a Datumaro dataset by merging datasets in Datumaro format.

        :return: Merged Datumaro dataset.
        """
        source_datasets = self._create_source_datasets()
        return HLOps.merge(*source_datasets, merge_policy="union")

    def _create_source_datasets(self) -> list:
        """
        Create a list of Dataset objects from tasks in Datumaro format.

        :return: List of Datumaro datasets.
        """
        return [Dataset.import_from(os.path.join(self.datasets_path, name), 'datumaro') for name in os.listdir(self.datasets_path)]

    def _mapping_labels(self, dataset: Dataset, mapping: list) -> Dataset:
        """
        Map labels from source to target.

        :param dataset: Datumaro dataset.
        :param mapping: List of tuples for label mapping. Example: [('source_label', 'target_label')].

        :return: Mapped Datumaro dataset.
        """
        return dataset.transform('remap_labels', mapping=mapping)

    def _reindex_labels(self, dataset: Dataset, labels_id_mapping: dict) -> Dataset:
        """
        Reindex labels in the dataset.

        :param dataset: Datumaro dataset.
        :param labels_id_mapping: Dict mapping label names to target IDs.

        :return: Reindexed Datumaro dataset.
        """
        category_dict_name_ind = dataset.categories().get(AnnotationType.label)._indices
        category_dict_ind_name = {v: k for k, v in category_dict_name_ind.items()}

        for item in dataset:
            for annotation in item.annotations:
                annotation.label = labels_id_mapping[category_dict_ind_name[annotation.label]]
                
        dataset.categories().get(AnnotationType.label)._indices = labels_id_mapping

        return dataset

    def _export_coco(self, dataset: Dataset, path: str) -> None:
        """
        Export dataset in COCO format.

        :param dataset: Datumaro dataset.
        :param path: Path to save the dataset.
        """
        exporter = dataset.env.exporters['coco']
        exporter._TASK_CONVERTER[CocoTask.instances] = PatchInstancesExporter
        dataset.export(save_dir=path, format=exporter, save_media=True)

    def _export_yolo(self, dataset: Dataset, path: str) -> None:
        """
        Export dataset in YOLO format.

        :param dataset: Datumaro dataset.
        :param path: Path to save the dataset.
        """
        self._export_coco(dataset, path)

        use_segments = "seg" in self.export_format
        coco2yolo = COCOConverter(
            json_dir=os.path.join(path, 'annotations'),
            save_dir=os.path.join(path, 'labels'),
            use_segments=use_segments,
            convert_format='yolo'
        )

        coco2yolo.convert()
        os.system(f'rm -rf {path}/annotations')

    def export_dataset(self, save_path: str = None) -> None:
        """
        Transform and save the dataset in the specified format.

        :param save_path: Path to save the transformed dataset.
        """
        if self.splits:
            self.dataset = self.dataset.transform('random_split', splits=self.splits)
        
        if self.mapping:
            self.dataset = self._mapping_labels(self.dataset, self.mapping)

        if self.labels_id_mapping:
            self.dataset = self._reindex_labels(self.dataset, self.labels_id_mapping)
        
        if not save_path or save_path == self.datasets_path:
            save_path = f'{self.datasets_path}_{self.export_format}'

        if 'yolo' in self.export_format:
            self._export_yolo(self.dataset, save_path)
        elif 'coco' in self.export_format:
            self._export_coco(self.dataset, save_path)
        else:
            self.dataset.export(save_dir=save_path, format=self.export_format, save_media=True)

        return save_path