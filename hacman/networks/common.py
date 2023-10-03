import torch
from hacman.networks.point_transformer import PointTransformerSegmentation
from hacman.networks.pointnet2 import PointNet2Classification, PointNet2Segmentation
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_model_config(parser):
    parser.add_argument("--model", default="pn", type=str, help="Model Name")
    parser.add_argument("--fps_deterministic", action="store_true", help="Turn off fps random start")
    parser.add_argument("--pos_in_feature", action="store_true", help="Include absolute point position in feature")
    parser.add_argument("--dropout", default=0., type=float, help="dropout")
    parser.add_argument("--normalize_pos", default=False, action='store_true')
    return

def init_network(config, input_channels, output_channels, mode=None):
    if config["model"] == 'pt':
        network = PointTransformerSegmentation(input_channels, output_channels, 
                                            dim_model=[32, 64, 128, 256, 512], 
                                            k=16, 
                                            fps_random_start=not config['fps_deterministic'],
                                            pos_in_feature=config["pos_in_feature"],
                                            normalize_pos=config["normalize_pos"]).to(device)
    elif config["model"] == 'pn':
        network = PointNet2Segmentation(input_channels, output_channels, 
                                    dropout=config['dropout'],
                                    fps_random_start=not config['fps_deterministic'],
                                    pos_in_feature=config["pos_in_feature"],
                                    normalize_pos=config["normalize_pos"],
                                    normalize_pos_param=config["normalize_pos_param"]).to(device)
    elif config['model'] == 'pnc':
        network = PointNet2Classification(input_channels, out_channels=output_channels, 
                dropout=config['dropout'],
                fps_random_start=not config['fps_deterministic'],
                pos_in_feature=config["pos_in_feature"],
                normalize_pos=config["normalize_pos"],
                normalize_pos_param=config["normalize_pos_param"],
                mode=mode).to(device)

    else:
        print(f"Model '{config['model']}' does not exist.")
        assert NotImplementedError
    return network

def load_network(network, load_path):
    loaded = torch.load(load_path, map_location=device)
    if 'model_state_dict' in loaded.keys():
        network.load_state_dict(loaded['model_state_dict'])
    else:
        # Backward compatibility
        network.load_state_dict(loaded)
    return network
