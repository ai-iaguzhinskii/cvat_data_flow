import os
from collections import OrderedDict
import json
import logging

from datumaro.components.project import Project
from datumaro.components.operations import IntersectMerge
from datumaro.components.errors import QualityError, MergeError


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
        export_format='coco'
    )
    dataset.transform_dataset(
        splits=[('train', 0.7), ('val', 0.2), ('test', 0.1)]
    )
    ```
    As a result, the dataset will be saved in '/path/to/datasets_coco_split' in the COCO format and will be divided into train, val and test subsets.
    """ 

    def __init__(self, datasets_path:str, export_format:str='coco'):
        """
        :param datasets_path: path to folder with cvat tasks in datumaro format
        :param export_format: final format of dataset
        """
        self.export_format = export_format
        self.datasets_path = datasets_path
        self.datasets_names = os.listdir(datasets_path)
        self.logger = logging.getLogger(__name__)

    def _create_projects(self) -> list:
        """
        Create list of datumaro.datasets from tasks in datumaro format
        """
        source_datasets = []
        for name in self.datasets_names:
            project = Project().import_from(path=os.path.join(self.datasets_path, name),
                                            dataset_format='datumaro')
            source_datasets.append(project.make_dataset())
        return source_datasets
    
    def _merge_datasets(self, source_datasets:list=None, export:bool=False):
        """
        Build dataset from tasks without spliting

        :param source_datasets: list of datumaro.datasets, see self._create_projects()
        :param export: if need save dataset in target format without spliting on train, val and test

        :return merger_project: datumaro.project
        :return merger_dataset: datumaro.dataset
        :return merger: datumaro.extractor
        """
        
        if source_datasets is None:
            source_datasets = self._create_projects()
        merger = IntersectMerge(conf=IntersectMerge.Conf(pairwise_dist=0.5, 
                                                         groups=[], 
                                                         output_conf_thresh=0.0, 
                                                         quorum=0)
        )
        merged_dataset = merger(source_datasets)
        merger_project = Project()
        output_dataset = merger_project.make_dataset()
        output_dataset.define_categories(merged_dataset.categories())
        merged_dataset = output_dataset.update(merged_dataset)

        if export:
            merged_dataset.export(save_dir=f'{self.datasets_path}_{self.format}', 
                                  format=self.export_format, 
                                  save_images=True
            )

        return merger_project, merged_dataset, merger
    
    def _mapping_labels(self, dataset: Project, project: Project, mapping: list):
        """
        Mapping labels from source to target
        
        :param dataset: datumaro.dataset
        :param mapping: list of tuple of mapping discription: ('source_label': 'target_label').

        :return filter_dataset: datumaro.dataset
        """
        
        method_mapping = project.env.make_transform('remap_labels')
        extractor_mapping = dataset.transform(method=method_mapping, mapping=mapping, default='delete')

        mapping_dataset = Project().make_dataset()
        mapping_dataset._categories = extractor_mapping.categories()
        mapping_dataset.update(extractor_mapping)

        filter_extractor = mapping_dataset.filter(expr='/item/annotation', filter_annotations=True, remove_empty=True)

        filter_dataset = Project().make_dataset()
        filter_dataset._categories = filter_extractor.categories()
        filter_dataset.update(filter_extractor)

        return filter_dataset
    
    def transform_dataset(self, splits: list, mapping: list = None, project=None, dataset=None, merger=None):
        """
        Random split dataset on subsests and filter image without annotations and save it.

        :param splits: list of tuple of spliting discription: ('subset_name': part_of_subset:float). 
        :param mapping: list of tuple of mapping discription: ('source_label': 'target_label'). Default: None
        :param project: datumaro.project, see self._merge_datasets()
        :param dataset: datumaro.dataset, see self._merge_datasets()
        :param merger: datumaro.extractor, see self._merge_datasets()
        """

        if dataset is None or project is None or merger is None:
            project, dataset, merger = self._merge_datasets()
        
        method_split = project.env.make_transform('random_split')
        extractor_split = dataset.transform(method=method_split, splits=splits)

        transform_dataset = Project().make_dataset()
        transform_dataset._categories = extractor_split.categories()
        transform_dataset.update(extractor_split)

        if mapping is not None:
            transform_dataset = self._mapping_labels(transform_dataset, project, mapping)

        transform_dataset.export(f'{self.datasets_path}_{self.export_format}_split', self.export_format, save_images=True)
        self.logger.info(f'Dataset saved in "{self.datasets_path}_{self.export_format}_split"')

        report_path = os.path.join(f'{self.datasets_path}_{self.export_format}_split', 'merge_report.json')
        self._save_merge_report(merger, report_path)
    
    
    @staticmethod
    def _save_merge_report(merger, path):
        item_errors = OrderedDict()
        source_errors = OrderedDict()
        all_errors = []

        for e in merger.errors:
            if isinstance(e, QualityError):
                item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1
            elif isinstance(e, MergeError):
                for s in e.sources:
                    source_errors[s] = source_errors.get(s, 0) + 1
                item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1

            all_errors.append(str(e))

        errors = OrderedDict([
            ('Item errors', item_errors),
            ('Source errors', source_errors),
            ('All errors', all_errors),
        ])

        with open(path, 'w') as f:
            json.dump(errors, f, indent=4)
        

            