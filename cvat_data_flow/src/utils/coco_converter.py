"""
Description: Convert COCO JSON to YOLO format with segmentation support.
"""
import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class COCOConverter:
    """
    Convert COCO to new format.

    Main methods:
    - convert

    Convert COCO JSON to YOLO format with segmentation support:

    Example:
    ```
    coco2yolo = COCOConverter(
        json_dir='data/coco/annotations',
        save_dir='./converted_labels/',
        use_segments=True,
        convert_format='yolo'
    )

    coco2yolo.convert()
    """
    def __init__(self, json_dir: str, save_dir: str, use_segments: bool = True, convert_format: str = 'yolo'):
        """
        Initialize the COCOConverter object.

        :param json_dir: The path to the COCO JSON directory.
        :param save_dir: The path to save the converted labels.
        :param use_segments: Flag indicating whether to use segmentation.
        :param convert_format: The format to convert to.
        """
        self.json_dir = json_dir
        self.save_dir = save_dir
        self.convert_format = convert_format
        self.use_segments = use_segments

    def _min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> tuple:
        """
        Find a pair of indexes with the shortest distance.

        :param arr1: The first array.
        :param arr2: The second array.

        :return: A pair of indexes with the shortest distance.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def _merge_multi_segment(self, segments: list) -> list:
        """
        Merge multiple segments into one.

        :param segments: The segments to merge.

        :return: The merged segments.
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        for i in range(1, len(segments)):
            idx = self._min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx[0])
            idx_list[i].append(idx[1])

        for k in range(2):
            if k == 0:
                for i, idx in enumerate(idx_list):
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s

    def _bbox_process(self, bbox: list, img_dimensions: tuple) -> list:
        """
        Process a single annotation into a bbox format.

        :param bbox: The bbox to process.
        :param img_dimensions: The dimensions of the image.

        :return: The processed bbox.
        """
        h, w = img_dimensions
        box = np.array(bbox, dtype=np.float64)
        box[:2] += box[2:] / 2
        box[[0, 2]] /= w
        box[[1, 3]] /= h

        return box.tolist()
    
    def _process_annotation(self, ann: dict, img_dimensions: tuple) -> tuple:
        """
        Process a single annotation into a bbox or segment format.

        :param ann: The annotation to process.
        :param img_dimensions: The dimensions of the image.

        :return: The processed bbox or segment.
        """
        h, w = img_dimensions
        if ann['iscrowd']:
            return None, None
        
        box = self._bbox_process(ann['bbox'], img_dimensions)
        if box[2] <= 0 or box[3] <= 0:  # Skip invalid boxes
            return None, None

        cls = ann['category_id']
        box = [cls] + box

        segment = None
        if self.use_segments and 'segmentation' in ann:
            if len(ann['segmentation']) > 1:
                merged_segment = self._merge_multi_segment(ann['segmentation'])
                segment = (np.concatenate(merged_segment, axis=0) / np.array([w, h])).reshape(-1).tolist()
            else:
                segment = [j for i in ann['segmentation'] for j in i]
                segment = (np.array(segment).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
            segment = [cls] + segment

        return box, segment
    
    def _to_yolo(self):
        """
        Convert COCO JSON to YOLO format
        """

        # Create the save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Get only instance segmentation COCO JSONs and sort them
        sub_dataset_json_paths = [os.path.join(self.json_dir, json_file) for json_file in os.listdir(self.json_dir) if "instances_" in json_file and json_file.endswith('.json')]
        for json_file in sub_dataset_json_paths:

            # Create a sub dataset directory to save the converted labels(ususally the same as the json file name: train, val, test)
            sub_dataset_path = os.path.join(self.save_dir, os.path.basename(json_file).split('_')[-1].split('.')[0])
            os.makedirs(sub_dataset_path, exist_ok=True)

            # Load COCO JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            images = {img["id"]: img for img in data['images']}
            img_to_anns = defaultdict(list)
            for ann in data['annotations']:
                img_to_anns[ann['image_id']].append(ann)

            for img_id, anns in tqdm(img_to_anns.items(), desc=f'Annotations {json_file}'):
                img = images[img_id]
                h, w, f = img['height'], img['width'], img['file_name']
                img_dimensions = (h, w)

                bboxes, segments = [], []
                for ann in anns:
                    box, segment = self._process_annotation(ann, img_dimensions)
                    if box:
                        bboxes.append(box)
                    if segment:
                        segments.append(segment)

                with open(os.path.join(sub_dataset_path, f'{img_id}.txt'), 'w') as file:
                    for box_or_seg in (segments if self.use_segments else bboxes):
                        line = ' '.join(map(str, box_or_seg))
                        file.write(line + '\n')

    def convert(self):
        """
        Convert COCO JSON to the specified format.
        """
        
        if self.convert_format == 'yolo':
            self._to_yolo()
        else:
            raise ValueError(f'Unknown convert format: {self.convert_format}.')
        

# if __name__ == '__main__':
#     coco2yolo = COCOConverter(
#         json_dir='data/astra_datasets/project_64_coco_split/annotations',
#         save_dir='./converted_labels/',
#         use_segments=True,
#     )
#     coco2yolo.convert()
