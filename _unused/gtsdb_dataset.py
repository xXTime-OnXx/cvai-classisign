import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Callable, Union


class GTSDBDataset(Dataset):
    """
    PyTorch Dataset for the German Traffic Sign Detection Benchmark (GTSDB).
    
    This dataset handles the GTSDB which contains full images with bounding boxes
    for traffic signs categorized into prohibitory, mandatory, danger, and other categories.
    
    Attributes:
        root_dir (str): Directory with all the images and annotation files
        split (str): Dataset split ('train' or 'test')
        transform (callable, optional): Optional transform to be applied to the image
        target_transform (callable, optional): Optional transform to be applied to the targets
        class_map (dict): Mapping from class IDs to readable names
        category_map (dict): Mapping from class IDs to category IDs
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_image_path: bool = False,
        train_dir_name: str = 'TrainIJCNN2013',  # Add these parameters
        test_dir_name: str = 'TestIJCNN2013',    # to make it configurable
    ):
        """
        Args:
            root_dir (str): Directory with all the images and annotation files
            split (str): Dataset split ('train' or 'test')
            transform (callable, optional): Optional transform to be applied to the image
            target_transform (callable, optional): Optional transform to be applied to the targets
            return_image_path (bool): Whether to return the image path with the sample
        """
        self.root_dir = root_dir
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.return_image_path = return_image_path
        
        # Verify split is valid
        if self.split not in ['train', 'test']:
            raise ValueError(f"Split must be either 'train' or 'test', got {self.split}")
            
        # Define the paths
        if self.split == 'train':
            self.data_dir = os.path.join(root_dir, train_dir_name + '/' + train_dir_name)
        else:  # test
            self.data_dir = os.path.join(root_dir, test_dir_name + '/' + 'TestIJCNN2013Download')
        
        # Define class mappings and categories
        self._init_class_mappings()
        
        # Load the annotations
        self.annotations = self._load_annotations()
    
    def _init_class_mappings(self):
        """Initialize the class and category mappings for the GTSDB dataset."""
        
        # Class mapping (class ID to human-readable name)
        # This is a partial class mapping - extend as needed
        self.class_map = {
            0: 'Speed limit 20',
            1: 'Speed limit 30',
            2: 'Speed limit 50',
            3: 'Speed limit 60',
            4: 'Speed limit 70',
            5: 'Speed limit 80',
            7: 'End of speed limit 80',
            8: 'Speed limit 100',
            9: 'Speed limit 120',
            10: 'No passing',
            11: 'No passing for vehicles over 3.5t',
            12: 'Right-of-way at the next intersection',
            13: 'Priority road',
            14: 'Yield',
            15: 'Stop',
            16: 'No vehicles',
            17: 'No vehicles over 3.5t',
            18: 'No entry',
            19: 'General caution',
            20: 'Dangerous curve to the left',
            21: 'Dangerous curve to the right',
            22: 'Double curve',
            23: 'Bumpy road',
            24: 'Slippery road',
            25: 'Road narrows on the right',
            26: 'Road work',
            27: 'Traffic signals',
            28: 'Pedestrians',
            29: 'Children crossing',
            30: 'Bicycles crossing',
            31: 'Beware of ice/snow',
            32: 'Wild animals crossing',
            33: 'End of all speed and passing limits',
            34: 'Turn right ahead',
            35: 'Turn left ahead',
            36: 'Ahead only',
            37: 'Go straight or right',
            38: 'Go straight or left',
            39: 'Keep right',
            40: 'Keep left',
            41: 'Roundabout mandatory',
            42: 'End of no passing',
            43: 'End of no passing for vehicles over 3.5t'
        }
        
        # Category mappings (class ID to category)
        # Based on the dataset description:
        # prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16] (circular, white ground with red border)
        # mandatory = [33, 34, 35, 36, 37, 38, 39, 40] (circular, blue ground)
        # danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] (triangular, white ground with red border)
        # other = [6, 12, 32, 41, 42]
        
        # Define category labels
        self.categories = {
            0: 'prohibitory',
            1: 'mandatory',
            2: 'danger',
            3: 'other'
        }
        
        # Map from class ID to category ID
        self.category_map = {}
        
        # Prohibitory signs
        for class_id in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]:
            self.category_map[class_id] = 0
            
        # Mandatory signs
        for class_id in [33, 34, 35, 36, 37, 38, 39, 40]:
            self.category_map[class_id] = 1
            
        # Danger signs
        for class_id in [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
            self.category_map[class_id] = 2
            
        # Other signs
        for class_id in [6, 12, 32, 41, 42]:
            self.category_map[class_id] = 3
    
    def _load_annotations(self) -> List[Dict]:
        """
        Load annotations from the CSV file.
        
        Returns:
            List[Dict]: List of dictionaries with annotations for each image
        """
        annotations = []
        
        if self.split == 'train':
            # For training set, we need to load the gt.txt file
            gt_file = os.path.join(self.data_dir, 'gt.txt')
            with open(gt_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    # Each row format: Filename;left;top;right;bottom;ClassID
                    filename = row[0]
                    left = int(float(row[1]))
                    top = int(float(row[2]))
                    right = int(float(row[3]))
                    bottom = int(float(row[4]))
                    class_id = int(row[5])
                    
                    # Get the category from the class
                    category_id = self.category_map.get(class_id, 3)  # Default to "other" category if not found
                    
                    # Add to annotations
                    annotations.append({
                        'filename': filename,
                        'bbox': [left, top, right, bottom],
                        'class_id': class_id,
                        'category_id': category_id
                    })
        else:
            # For test set, we need to load the gt.txt file if available
            gt_file = os.path.join(self.data_dir, 'gt.txt')
            if os.path.exists(gt_file):
                with open(gt_file, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    for row in reader:
                        # Same format as training
                        filename = row[0]
                        left = int(float(row[1]))
                        top = int(float(row[2]))
                        right = int(float(row[3]))
                        bottom = int(float(row[4]))
                        class_id = int(row[5])
                        
                        # Get the category from the class
                        category_id = self.category_map.get(class_id, 3)
                        
                        # Add to annotations
                        annotations.append({
                            'filename': filename,
                            'bbox': [left, top, right, bottom],
                            'class_id': class_id,
                            'category_id': category_id
                        })
            else:
                # If gt.txt doesn't exist for test, list all images
                image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.ppm')]
                for filename in image_files:
                    annotations.append({
                        'filename': filename,
                        'bbox': [0, 0, 0, 0],  # Default bbox
                        'class_id': -1,  # No class available
                        'category_id': -1  # No category available
                    })
        
        return annotations

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.
        
        Returns:
            int: Length of the dataset
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Union[Tuple, Dict]:
        """
        Get the image and target for a specific index.
        
        Args:
            idx (int): Index to get the sample for
            
        Returns:
            tuple: (image, target) where target is a dictionary with annotation info
            or dict: {'image': image, 'target': target, 'path': image_path} if return_image_path is True
        """
        ann = self.annotations[idx]
        filename = ann['filename']
        image_path = os.path.join(self.data_dir, filename)
        
        # Load the image
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        # Create the target
        target = {
            'boxes': torch.tensor([ann['bbox']], dtype=torch.float32),
            'labels': torch.tensor([ann['class_id']], dtype=torch.int64),
            'category_id': torch.tensor([ann['category_id']], dtype=torch.int64)
        }
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        if self.return_image_path:
            return {'image': img, 'target': target, 'path': image_path}
        else:
            return img, target

    def get_class_name(self, class_id: int) -> str:
        """
        Get the human-readable class name for a given class ID.
        
        Args:
            class_id (int): Class ID to get the name for
            
        Returns:
            str: Class name or 'Unknown' if not found
        """
        return self.class_map.get(class_id, 'Unknown')
    
    def get_category_name(self, category_id: int) -> str:
        """
        Get the category name for a given category ID.
        
        Args:
            category_id (int): Category ID to get the name for
            
        Returns:
            str: Category name or 'Unknown' if not found
        """
        return self.categories.get(category_id, 'Unknown')