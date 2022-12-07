"""
Run MC Dropout on a single image
"""

import torch
from PIL import Image
from pathlib import Path

import predict
from datasets import Cityscapes

def main():
    R = 10 # Number of times to run the MC Dropout loop
    input = str(Path('/datasets', 'cityscapes', 'leftImg8bit', 'test', 'berlin', 'berlin_000000_000019_leftImg8bit.png'))
    model = str(Path('checkpoints', 'best_deeplabv3plus_resnet101_cityscapes_os16.pth'))
    output_dir = str(Path('test_images'))

    input_list = input.split('/')
    for i in range(R):
        output = predict.main(['--input', input, '--dataset', 'cityscapes', '--model', 'deeplabv3plus_resnet101', '--ckpt', model, '--save_val_results_to', output_dir, '--new_implement', 'True', '--mc_dropout', 'True', '--input_num', str(i)])
        print(output) # print for sanity check (dropout is enabled during eval)
        if i == 0:
            output_sum = output
        else:
            output_sum = torch.add(output_sum, output)

    output_MC = output_sum/R
    pred_MC = output_MC.max(1)[1].cpu().numpy()[0]
    colorized_pred_MC = Cityscapes.decode_target(pred_MC).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
    colorized_pred_MC = Image.fromarray(colorized_pred_MC)
    colorized_pred_MC.save(str(Path(output_dir, input_list[len(input_list)-1])))

if __name__ == '__main__':
    main()
