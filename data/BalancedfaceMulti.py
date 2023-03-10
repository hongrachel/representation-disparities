import os
import cv2
import csv
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T


class BalancedfaceMulti(data.Dataset):

    def __init__(self, root, identities_per_race, images_per_identity=None, input_shape=(3, 128, 128), face_ratios_file=None, seed=0):
        self.face_ratios_file = face_ratios_file

        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(int(self.input_shape[1] * 156 / 128)),
            T.RandomCrop(self.input_shape[1:]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])
        self.img_paths = []
        self.labels = []

        random.seed(42 + seed)

        african_dir = os.path.join(root, 'African')
        asian_dir = os.path.join(root, 'Asian')
        caucasian_dir = os.path.join(root, 'Caucasian')
        indian_dir = os.path.join(root, 'Indian')

        if images_per_identity is None:
            images_per_label = 18
            print('images_per_label', images_per_label)
            images_per_identity = {}
            images_per_identity['african'] = images_per_label
            images_per_identity['asian'] = images_per_label
            images_per_identity['caucasian'] = images_per_label
            images_per_identity['indian'] = images_per_label

        african_labels_to_images, african_labels = self.get_labels(african_dir, images_per_identity['african'])
        asian_labels_to_images, asian_labels = self.get_labels(asian_dir, images_per_identity['asian'])
        caucasian_labels_to_images, caucasian_labels = self.get_labels(caucasian_dir, images_per_identity['caucasian'])
        indian_labels_to_images, indian_labels = self.get_labels(indian_dir, images_per_identity['indian'])

        cur_label = 0
        for label in african_labels[:identities_per_race['african']]:
            self.img_paths.extend(african_labels_to_images[label])
            self.labels.extend([cur_label] * images_per_identity['african'])
            cur_label += 1
        for label in asian_labels[:identities_per_race['asian']]:
            self.img_paths.extend(asian_labels_to_images[label])
            self.labels.extend([cur_label] * images_per_identity['asian'])
            cur_label += 1
        for label in caucasian_labels[:identities_per_race['caucasian']]:
            self.img_paths.extend(caucasian_labels_to_images[label])
            self.labels.extend([cur_label] * images_per_identity['caucasian'])
            cur_label += 1
        for label in indian_labels[:identities_per_race['indian']]:
            self.img_paths.extend(indian_labels_to_images[label])
            self.labels.extend([cur_label] * images_per_identity['indian'])
            cur_label += 1

    def get_paths(self, root_dir):
        paths = None
        if self.face_ratios_file:
            paths = []
            with open(self.face_ratios_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                row_num = 0
                header = []
                for row in reader:
                    path = row[0].split(',')[1]
                    if row_num != 0 and path.startswith(root_dir):
                        paths.append(path)
                    row_num += 1
            paths = set(paths)
        return paths

    def get_labels(self, root_dir, num_images):
        filtered_paths = self.get_paths(root_dir)

        label_to_images = {}
        cur_label = 0
        for dir in os.listdir(os.path.join(root_dir)):
            paths = []
            cur_num_images = 0
            for img in os.listdir(os.path.join(root_dir, dir)):
                if filtered_paths is None:
                    path = os.path.join(root_dir, dir, img)
                    paths.append(path)
                    cur_num_images += 1
                else:
                    path = os.path.join(root_dir, dir, img)
                    if path in filtered_paths:
                        paths.append(path)
                        cur_num_images += 1
            
            if cur_num_images >= num_images:
                label_to_images[cur_label] = random.sample(paths, num_images)
                cur_label += 1
        
        labels = list(label_to_images.keys())
        random.shuffle(labels)

        # assert(len(labels) >= 5000)
        return label_to_images, labels


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)
