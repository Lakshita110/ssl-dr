import os
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

class traindataset(data.Dataset):

    def __init__(self, root, transform=None, train=True, args=None):
        """
        Args:
            root (str): Root directory containing the dataset.
            transform (callable, optional): Transformations to be applied to the images.
            train (bool): Whether to load the training dataset or the test dataset.
            args (Namespace): Additional arguments for multitask, augmentation, etc.
        """
        self.root_dir = root
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),  # Resize images to 320x320
            transforms.ToTensor(),          # Convert to PyTorch tensor
        ])
        self.name = []
        self.train = train
        self.multitask = args.multitask
        self.multiaug = args.multiaug

        if self.train:
            # Load training data paths
            train_path = os.path.join(self.root_dir, "test/test.txt")
            self.train_dataset = [os.path.join(self.root_dir, "test", line.strip()) 
                                for line in np.genfromtxt(train_path, dtype=str)]
            self.targets = [0] * len(self.train_dataset)  # No human-annotated labels for training
            self.rotation_label = [0] * len(self.train_dataset)
            
            self.train_dataset = self.train_dataset[:500]
            self.targets = self.targets[:500]
            self.rotation_label = self.rotation_label[:500]
    
        else:
            self.train_dataset = []
            self.targets = []
            self.name = []
            
            # Load test data
            test_path = os.path.join(self.root_dir, "Training/train.txt")
            label_file = os.path.join(self.root_dir, "Training/image_labels.txt")
            test_paths = list(np.genfromtxt(test_path, dtype=str))
            test_labels = np.loadtxt(label_file, dtype='uint8')

            for i in range(len(test_paths)):
                self.train_dataset.append(os.path.join(self.root_dir, "Training", test_paths[i]))
                self.targets.append(test_labels[i])
                self.name.append(test_paths[i])

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        
        Returns:
            img (Tensor): Transformed image.
            target (int): Target label.
            idx (int): Index of the sample.
            name (str): Name of the image.
        """
        # Get the image path and load the image
        img_path = self.train_dataset[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(np.uint8(img))  # Convert to PIL Image
        target = self.targets[idx]

        if self.transform:
            img1 = self.transform(img)
            
            # if self.train and self.multiaug:
            #     img2 = self.transform(img)
            #     images = [img1, img2]
                
            #     if self.multitask:
            #         targets = [target, self.rotation_labels[idx]]
            #     else:
            #         targets = target
                    
            #     return images, targets, idx, 0
            
            return img1, target, idx, 0
        
        return img, target, idx, 0


if __name__ == '__main__':
    # Example: Initialize dataset
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize images to 320x320
        transforms.ToTensor(),          # Convert to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (example for grayscale)
    ])

    args = type('Args', (), {'multitask': False, 'multiaug': False, 'synthesis': False})

    # Root directory where the dataset is stored
    root_dir = "/cluster/tufts/cs152l3dclass/ljain01"
    test_root_dir = "/cluster/tufts/cs152l3dclass/areddy05/IDRID/Images"

    # Load training dataset
    train_dataset = traindataset(root=root_dir, transform=transform, train=True, args=args)

    # Load test dataset
    test_dataset = traindataset(root=test_root_dir, transform=transform, train=False, args=args)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")