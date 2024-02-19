import os
import shutil
import zipfile
import json
import logging  

from cvat_sdk import make_client
from typing import Dict, Any
from tqdm import tqdm

class CVATUploader:
    """
    A class used to upload datasets from CVAT in Datumaro format.

    Main methods:
    - upload_tasks_from_cvat
    - upload_projects_from_cvat

    Download and extract dataset from CVAT, rename images and update annotation file in Datumaro format and save it to the specified directory:

    Example:
    ```
    cvat_uploader = CVATUploader(url='http://cvat.example.com',
                                    login='username',
                                    password='password',
                                    save_path='/path/to/save/dataset')
    cvat_uploader.upload_projects_from_cvat(project_ids=[1, 2, 3])
    ```
    """

    # Number of images in the dataset
    img_num = 0

    def __init__(self, url: str, login: str, password: str, save_path: str):
        """
        :param url: CVAT url.
        :param login: CVAT login.
        :param password: CVAT password.
        :param save_path: Path to the directory where the dataset will be saved.
        :param logger: Logger.
        """

        logging.basicConfig(level=logging.INFO)
        logging.getLogger('cvat_sdk.core.client').setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f'Connecting to CVAT "{url}" ...')
        self.client = make_client(host=url, credentials=(login, password))
        self.downloads_directory = save_path
        os.makedirs(self.downloads_directory, exist_ok=True)

    def _preroc_data(self, path: str) -> int:
        """
        Rename images and update annotation file.

        :param path: Path to the directory with images and annotation file.
        :return: The number of the next image in the dataset.  
        """

        annotation_path = os.path.join(path, 'annotations', 'default.json')
        images_path = os.path.join(path, 'images')

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        for i, item in enumerate(data['items']):
            old_image_path = os.path.join(images_path, item['image']['path'])
            new_image_path = os.path.join(images_path, f'{self.img_num}.jpg')
            os.rename(old_image_path, new_image_path)

            data['items'][i]['image']['path'] = f'{self.img_num}.jpg'
            data['items'][i]['id'] = str(self.img_num)
            data['items'][i]['media']['path'] = f'{self.img_num}.jpg'

            self.img_num += 1

        with open(annotation_path, 'w') as f:
            json.dump(data, f)

    def _download_and_extract(self, id: int, type: str):
        """
        Download and extract dataset from CVAT.

        :param id: Dataset id.
        :param type: Dataset type (task or project).
        """

        archive_path = os.path.join(self.downloads_directory, str(id) + '.zip')

        if os.path.exists(archive_path):
            os.remove(archive_path)

        self.logger.debug(f'downloading "{archive_path}" ...')

        try:
            if type == "task":
                info = self.client.tasks.retrieve(int(id))
            elif type == "project":
                info = self.client.projects.retrieve(int(id))

            info.export_dataset('Datumaro 1.0', archive_path)
        except Exception as e:
            self.logger.error(f'failed to download "{archive_path}": {e}')
            return

        self.logger.debug(f'successfully downloaded "{archive_path}"')

        target_directory = os.path.join(self.downloads_directory, str(id))
        os.makedirs(target_directory, exist_ok=True)

        self.logger.debug(f'deflating "{archive_path}" to "{target_directory}" ...')

        try:
            with open(archive_path, 'rb') as zip_input:
                dataset_zip = zipfile.ZipFile(zip_input)
                dataset_zip.extractall(target_directory)

            source_images_directory = os.path.join(target_directory, 'images', 'default')
            target_images_directory = os.path.join(target_directory, 'images')

            for image in os.listdir(source_images_directory):
                full_image_path = os.path.join(source_images_directory, image)
                shutil.move(full_image_path, target_images_directory)

            shutil.rmtree(source_images_directory)

        except Exception as e:
            self.logger.error(f'failed to deflate "{archive_path}": {e}')
            os.remove(archive_path)
            return

        self.logger.debug(f'successfully deflated "{archive_path}"')
        os.remove(archive_path)

    def _get_task_ids(self, project_id: int) -> list:
        """
        Get task ids from project.

        :param project_id: Project id.
        :return: List of task ids.
        """

        project = self.client.projects.retrieve(project_id)
        task_ids = project.tasks

        self.logger.debug(f'tasks: {task_ids}')
        self.logger.info(f'Found {len(task_ids)} tasks in project "{project_id}"')

        return task_ids

    def upload_tasks_from_cvat(self, task_ids: list):
        """
        Upload tasks from CVAT.

        :param task_ids: List of task ids.
        """

        for task_id in tqdm(task_ids):
            self._download_and_extract(task_id, "task")
            target_directory = os.path.join(self.downloads_directory, str(task_id))
            self._preroc_data(target_directory)

    def upload_projects_from_cvat(self, project_ids: list):
        """
        Upload projects from CVAT.

        :param project_ids: List of project ids.
        """
        
        for project_id in tqdm(project_ids):
            task_ids = self._get_task_ids(int(project_id))

            self.logger.info(f'Downloading tasks from project "{project_id}"')
            self.upload_tasks_from_cvat(task_ids)
        
        self.logger.info(f'Finished uploading projects {project_ids} from CVAT. Final number of images: {self.img_num}')

        
