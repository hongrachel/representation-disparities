
from .config import Config

class RFWDataConfig(Config):

    def __init__(self):
        self.config = {
            'checkpoints_path': '~/RFW_checkpoints/',
            'features_dir': '~/RFW_feature_embeddings/',
            'train_root': '~/BUPT_dataset/race_per_7000/',
            'test_root': '~/RFW_dataset/',
            'identities_per_race_path': '~/RFW_images_per_race/',
            'rfw_test_list': '_pairs.txt',
            'simplex_file_path': 'configs/points_config_pair_500_to_10000.json',
        }
