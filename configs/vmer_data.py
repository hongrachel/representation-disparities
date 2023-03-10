from .config import Config

class VMERDataConfig(Config):

    def __init__(self):
        self.config = {
            'checkpoints_path': '~/VMER_checkpoints',
            'features_dir': '~/VMER_feature_embedddings/',
            'train_root': '~/VGG-Face2/data/train/',
            'test_root': '~/VGG-Face2/data/test/',
            'vmer_test_list_root': '~/VGG-Face2/data/',
            'vmer_test_list': '_pairs.csv',
            'balanced_train_json': '~/VGG-Face2/data/balanced_train.json',
            'balanced_test_json': '~/VGG-Face2/data/balanced_test.json',
            'simplex_file_path': 'configs/points_config_pair_50_to_800.json',
        }