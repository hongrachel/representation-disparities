import os
import json
import pandas as pd
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T


def getVMERList(pair_list_file):
    test_pairs = pd.read_csv(pair_list_file)

    pair_list = []
    pair_lines = test_pairs[['p1', 'p2', 'label']].values.tolist()
    for pair_line in pair_lines:
        p1 = pair_line[0]
        p2 = pair_line[1]
        label = int(pair_line[2])

        first_image = '/'.join(p1.split('/')[-2:])[:-4]
        second_image = '/'.join(p2.split('/')[-2:])[:-4]
        pair_list.append((first_image, second_image, label))

    return pair_list


class VMERTrain(data.Dataset):

    def __init__(self, root, identities_per_race, images_per_identity=None, input_shape=(3, 128, 128), balanced_train_json=None, seed=0):
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

        train_identities_by_race = {}
        with open(balanced_train_json, 'r') as f:
            train_identities_by_race = json.load(f)

        if images_per_identity is None:
            images_per_label = 108
            print('images_per_label', images_per_label)
            images_per_identity = {}
            images_per_identity['african'] = images_per_label
            images_per_identity['asian'] = images_per_label
            images_per_identity['caucasian'] = images_per_label
            images_per_identity['indian'] = images_per_label

        african_labels_to_images, african_labels = self.get_labels(root, train_identities_by_race['African'], images_per_identity['african'])
        asian_labels_to_images, asian_labels = self.get_labels(root, train_identities_by_race['Asian'], images_per_identity['asian'])
        caucasian_labels_to_images, caucasian_labels = self.get_labels(root, train_identities_by_race['Caucasian'], images_per_identity['caucasian'])        
        indian_labels_to_images, indian_labels = self.get_labels(root, train_identities_by_race['Indian'], images_per_identity['indian'])

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


    def get_labels(self, root, train_identities, num_images):
        label_to_images = {}
        cur_label = 0
        for dir in os.listdir(root):
            dir_path = os.path.join(root, dir)
            if os.path.isdir(dir_path) and dir in train_identities:
                paths = []
                cur_num_images = 0
                for img in os.listdir(dir_path):
                    paths.append(os.path.join(dir_path, img))
                    cur_num_images += 1

                if cur_num_images >= num_images:
                    label_to_images[cur_label] = random.sample(paths, num_images)
                    cur_label += 1

        labels = list(label_to_images.keys())
        random.shuffle(labels)
        assert(len(labels) >= 400)
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


class VMERTest(data.Dataset):

    def __init__(self, test_root, race='African', input_shape=(3, 128, 128), balanced_test_json=None):
        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(int(self.input_shape[1] * 128 / 128)),
            T.CenterCrop(self.input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])

        self.img_paths = []
        test_identities = []
        with open(balanced_test_json, 'r') as f:
            d = json.load(f)
            test_identities = d[race]

        for dir in os.listdir(test_root):
            if dir in test_identities:
                for img in os.listdir(os.path.join(test_root, dir)):
                    self.img_paths.append(os.path.join(test_root, dir, img))


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), '/'.join(img_path.split('/')[-2:])[:-4]


    def __len__(self):
        return len(self.img_paths)
