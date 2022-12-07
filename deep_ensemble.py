"""
Run deep ensemble
"""

from pathlib import Path

import train

def main():
    M = 5 # Size of the ensemble
    data_dir = str(Path('/datasets', 'cityscapes'))

    # Train the model M times with the pretrained weights as means and sampling different weights for each model
    for model in range(M):
        train.main(['--data_root', data_dir, '--dataset', 'cityscapes', '--num_classes', '19', '--model', 'deeplabv3plus_resnet101', '--model_num', str(model+1), '--deep_ensemble', 'True'])


if __name__ == '__main__':
    main()