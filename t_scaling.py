"""
Run temperature scaling on a single image
"""

import torch
from torch.utils import data
from pathlib import Path
from PIL import Image

import network
from datasets import Cityscapes
from utils import ext_transforms as et

class Temperature(torch.nn.Module):

    def __init__(self, model, device):
        super(Temperature, self).__init__()
        self.model = model
        self.device = device
        self.temp = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.model.to(self.device)
        model.train()

    def val_data(self, data_root, val_batch_size):
        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_dst = Cityscapes(root=data_root, split='val', transform=val_transform)
        val_loader = data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2)
        return val_loader

    def forward(self, input):
        output = self.model(input)
        return output

    def set_temp(self, val_data_root, num_epochs, lr, criterion):
        optimizer = torch.optim.Adam([self.temp], lr=lr)

        val_loader = self.val_data(val_data_root, val_batch_size=4)
        epochs = 0
        while True:
            epochs += 1
            # print(self.model.state_dict()) # Sanity check (model weights remain unchanged)
            print("Epoch Num: ", epochs)
            print("Temperature: ", self.temp.detach().cpu().numpy()[0])
            print("************************")
            for (images, labels) in val_loader:
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    output = self.forward(images)
                with torch.set_grad_enabled(True):
                    output = output / self.temp.to(self.device)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            if epochs >= num_epochs:
                return self.temp

def main():
    # Loading pretrained model
    model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=8)
    model.load_state_dict( torch.load( Path('models', 'best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar') )['model_state']  )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    num_epochs = 25
    lr = 0.001
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Optimizing temperature on the val set
    temp_obj = Temperature(model=model, device=device)
    val_data_root = str(Path('/datasets', 'cityscapes'))
    temp = temp_obj.set_temp(val_data_root, num_epochs, lr, criterion)
    print("Final optimized temperature: ", temp.detach().cpu().numpy()[0])

if __name__ == '__main__':
    main()