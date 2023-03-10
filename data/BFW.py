import os
import pandas as pd
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T

RACE_TO_LETTER_DICT = {
    'African': 'B',
    'Asian': 'A',
    'Caucasian': 'W',
    'Indian': 'I'
}

RACE_TO_FOLDER_DICT = {
    'African': 'black',
    'Asian': 'asian',
    'Caucasian': 'white',
    'Indian': 'indian'
}


TEST_FOLD = 5

def get_identities_from_folds(pair_list_file, race=None, is_test=True):
    pairs = pd.read_csv(pair_list_file)
    dataset_pairs = None
    if is_test and race is not None:
        dataset_pairs = pairs[
            (pairs['fold'] == TEST_FOLD) \
            & (pairs['e1'] == RACE_TO_LETTER_DICT[race]) \
            & (pairs['e2'] == RACE_TO_LETTER_DICT[race])
        ]
    elif not is_test and race is None:
        dataset_pairs = pairs[
            (pairs['fold'] != TEST_FOLD)
        ]
    
    identities = []
    if dataset_pairs is not None:
        identities += [ path.split('/')[1] for path in dataset_pairs['p1'].values]
        identities += [ path.split('/')[1] for path in dataset_pairs['p2'].values]
    return set(identities)


def getBFWList(pair_list_file, test_race):
    pairs = pd.read_csv(pair_list_file)
    test_pairs = pairs[
        (pairs['fold'] == TEST_FOLD) \
        & (pairs['e1'] == RACE_TO_LETTER_DICT[test_race]) \
        & (pairs['e2'] == RACE_TO_LETTER_DICT[test_race])
    ]

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


class BFWTrain(data.Dataset):

    def __init__(self, root, identities_per_race, images_per_identity=None, input_shape=(3, 128, 128), pair_list_file=None, seed=0):
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
        train_identities = get_identities_from_folds(pair_list_file, is_test=False)

        african_dirs = [os.path.join(root, 'black_females'), os.path.join(root, 'black_males')]
        asian_dirs = [os.path.join(root, 'asian_females'), os.path.join(root, 'asian_males')]
        caucasian_dirs = [os.path.join(root, 'white_females'), os.path.join(root, 'white_males')]
        indian_dirs = [os.path.join(root, 'indian_females'), os.path.join(root, 'indian_males')]

        african_labels_to_images, african_labels = self.get_labels(african_dirs, train_identities)
        asian_labels_to_images, asian_labels = self.get_labels(asian_dirs, train_identities)
        caucasian_labels_to_images, caucasian_labels = self.get_labels(caucasian_dirs, train_identities)
        indian_labels_to_images, indian_labels = self.get_labels(indian_dirs, train_identities)

        if images_per_identity is None:
            images_per_label = 25
            images_per_identity = {}
            images_per_identity['african'] = images_per_label
            images_per_identity['asian'] = images_per_label
            images_per_identity['caucasian'] = images_per_label
            images_per_identity['indian'] = images_per_label

        cur_label = 0
        for label in random.sample(african_labels, identities_per_race['african']):
            self.img_paths.extend(random.sample(african_labels_to_images[label], images_per_identity['african']))
            self.labels.extend([cur_label] * images_per_identity['african'])
            cur_label += 1
        for label in random.sample(asian_labels, identities_per_race['asian']):
            self.img_paths.extend(random.sample(asian_labels_to_images[label], images_per_identity['asian']))
            self.labels.extend([cur_label] * images_per_identity['asian'])
            cur_label += 1
        for label in random.sample(caucasian_labels, identities_per_race['caucasian']):
            self.img_paths.extend(random.sample(caucasian_labels_to_images[label], images_per_identity['caucasian']))
            self.labels.extend([cur_label] * images_per_identity['caucasian'])
            cur_label += 1
        for label in random.sample(indian_labels, identities_per_race['indian']):
            self.img_paths.extend(random.sample(indian_labels_to_images[label], images_per_identity['indian']))
            self.labels.extend([cur_label] * images_per_identity['indian'])
            cur_label += 1


    def get_labels(self, root_dirs, train_identities):
        label_to_images = {}
        cur_label = 0
        for root_dir in root_dirs:
            for dir in os.listdir(os.path.join(root_dir)):
                if os.path.isdir(os.path.join(root_dir, dir)) and dir in train_identities:
                    label_to_images[cur_label] = []
                    num_images = 0
                    for img in os.listdir(os.path.join(root_dir, dir)):
                        label_to_images[cur_label].append(os.path.join(root_dir, dir, img))
                        num_images += 1
                    assert(num_images == 25)
                    cur_label += 1

        return label_to_images, list(label_to_images.keys())


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)


class BFWTest(data.Dataset):

    def __init__(self, test_root, race='African', input_shape=(3, 128, 128), pair_list_file=None):
        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(int(self.input_shape[1] * 156 / 128)),
            T.CenterCrop(self.input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])

        test_identities = get_identities_from_folds(pair_list_file, race, is_test=True)
        self.img_paths = []

        for suffix in ['_females', '_males']:
            root = os.path.join(test_root, RACE_TO_FOLDER_DICT[race] + suffix)
            for dir in os.listdir(root):
                if dir in test_identities:
                    for img in os.listdir(os.path.join(root, dir)):
                        self.img_paths.append(os.path.join(root, dir, img))


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), '/'.join(img_path.split('/')[-2:])[:-4]


    def __len__(self):
        return len(self.img_paths)
