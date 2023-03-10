
from .config import Config

class BFWDataConfig(Config):

    def __init__(self):
        self.config = {
            'checkpoints_path': '~/BFW_checkpoints',
            'features_dir': '~/BFW_feature_embeddings/',
            'train_root': '~/BFW/bfw-cropped-aligned/',
            'test_root': '~/BFW/bfw-cropped-aligned/',
            'bfw_test_list': '~/BFW/bfw-v0.1.5-datatable.csv',
            'simplex_file_path': 'configs/points_config_increase_1_to_2_races_80_to_160.json',
        }
