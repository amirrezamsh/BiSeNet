# Imports and setup
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
import sys
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np



sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (update if needed)
weights_path = './checkpoints/model_final_v2_city.pth'
cityscapes_root = r"D:\semester_3\AML\project\datasets\cityscapes"

# Ensure output directory exists
os.makedirs("./checkpoints", exist_ok=True)



sys.path.append('BiSeNet/lib')
from lib.models.bisenetv2 import BiSeNetV2

# Define image transformations: resize, to-tensor, normalize

transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def pil_to_long_tensor(pic):
    label = torch.from_numpy(np.array(pic)).long()
    label[label == 255] = 19  # void in Cityscapes fine annotations
    label[label >= 20] = 19   # remap all invalid classes to void
    return label


target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
    pil_to_long_tensor,
])



def replace_conv_out(sequential: nn.Sequential, out_channels: int = 20):
    """
    Replace the last Conv2d in a nn.Sequential with a new one that has out_channels.
    """
    # Find the index of the last Conv2d
    conv_idx = None
    for i, m in enumerate(sequential):
        if isinstance(m, nn.Conv2d):
            conv_idx = i
    if conv_idx is None:
        raise ValueError("No Conv2d found in the Sequential!")
    
    old_conv: nn.Conv2d = sequential[conv_idx]
    new_conv = nn.Conv2d(old_conv.in_channels, out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=(old_conv.bias is not None))
    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
    if new_conv.bias is not None:
        nn.init.constant_(new_conv.bias, 0)
    
    sequential[conv_idx] = new_conv



def train() :

    parser = ArgumentParser()

    parser.add_argument('--epochnum',default= 1 , help = "Enter the number of epoch model is starting from")
    args = parser.parse_args()
    try:
        epochnum = int(args.epochnum)
    except ValueError:
        print("Error: Invalid epochnum argument")
        exit(1)

  # Load Cityscapes training dataset
    train_dataset = Cityscapes(
        root=cityscapes_root,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=target_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

  # Initialize BiSeNetV2 model with 20 output classes (19 + background)
    model = BiSeNetV2(n_classes=20, aux_mode='train')


    #print statements to check
    print(model.head.conv_out)

    

    #place this onlt if this is the first time running this file
    replace_conv_out(model.head.conv_out)
    replace_conv_out(model.aux2.conv_out)
    replace_conv_out(model.aux3.conv_out)
    replace_conv_out(model.aux4.conv_out)
    replace_conv_out(model.aux5_4.conv_out)


    
    
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        missing = []
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module.") and own_state[name.split("module.")[-1]].shape == param.shape:
                    own_state[name.split("module.")[-1]].copy_(param)
                elif 'module.'+ name in own_state.keys() and own_state["module."+name].shape == param.shape:
                    own_state['module.' + name].copy_(param)
                elif name not in own_state and f"module.{name}" not in own_state and name.split("module.")[-1] not in own_state:
                    missing.append(name)
            else:
                if own_state[name].shape == param.shape :
                    own_state[name].copy_(param)
                else :
                    missing.append(name)
        print(f"missing keys : {missing}")
        return model
    
    
    state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage, weights_only=False)

    model = load_my_state_dict(model, state_dict)
    
    model.to(device)

    print("✅Model weights updated sucssessfully")


    for name, param in model.named_parameters():
        if 'conv_out' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    print("✅Gradiation turned on only on interested layers")

  # Define loss and optimizer (only params with requires_grad=True will be updated)
  
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

  # Training loop (5 epochs)
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for images, targets in loop:
            images = images.to(device)            # shape: (B, 3, H, W)
            targets = targets.to(device)          # shape: (B, H, W), values in [0..19]


            targets = targets.squeeze(1)

            optimizer.zero_grad()

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # If model returns auxiliary outputs as a tuple/list, take the first (main) output
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # unique_values = torch.unique(targets)
            # print("Unique target values:", unique_values)

            # outputs: (B, 20, H, W)
            # targets[targets >= 20] = 255


            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
        print(f"Epoch {epoch+epochnum}/{epochnum+5}, Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"./checkpoints/bisenetv2_finetuned_epoch{epoch + epochnum}.pth")

    # Save checkpoint
    torch.save(model.state_dict(), "./checkpoints/bisenetv2_finetuned.pth")

if __name__ == '__main__':
    train()

