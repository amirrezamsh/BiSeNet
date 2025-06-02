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
from torch.optim.lr_scheduler import StepLR




sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (update if needed)
weights_path = './checkpoints/model_final_v2_city.pth'
cityscapes_root = '../cityscapes'
# cityscapes_root = r"D:\semester_3\AML\project\datasets\cityscapes"

# Ensure output directory exists
os.makedirs("./checkpoints", exist_ok=True)


sys.path.append('BiSeNet/lib')
from lib.models.bisenetv2 import BiSeNetV2

# Define image transformations: resize, to-tensor, normalize

mapping_20 = { 
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
    7: 1, 8: 2,
    9: 0, 10: 0,
    11: 3, 12: 4, 13: 5,
    14: 0, 15: 0, 16: 0,
    17: 6, 18: 0, 19: 7, 20: 8, 21: 9,
    22: 10, 23: 11, 24: 12, 25: 13, 26: 14,
    27: 15, 28: 16,
    29: 0, 30: 0,
    31: 17, 32: 18, 33: 19,
    255:0
}

# Create fast lookup table (for performance)
mapping_array = np.zeros(256, dtype=np.uint8)
for k, v in mapping_20.items():
    mapping_array[k] = v

# Transformation function for the label
def pil_to_mapped_tensor(pic):
    label = np.array(pic)  # Convert PIL to numpy
    # label[label == -1] = 255
    label = mapping_array[label]  # Apply mapping
    return torch.from_numpy(label).long()


transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])



target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
    pil_to_mapped_tensor,
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

  # Initialize BiSeNetV2 model with 20 output classes (19 + background)
    model = BiSeNetV2(n_classes=20, aux_mode='train')


    #print statements to check
    #print(model.head.conv_out)

    

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

    num_grad_true = 0
    for name, param in model.named_parameters():
        if 'conv_out' in name:
            param.requires_grad = True
            num_grad_true+=1
        else:
            param.requires_grad = False


    print(f"✅Gradiation turned on only on {num_grad_true} parameters")

  # Define loss and optimizer (only params with requires_grad=True will be updated)
  
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4, 
    weight_decay=1e-4  # small weight decay for regularization
)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

  # Training loop (5 epochs)
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for images, targets in loop:
            images = images.to(device)            # shape: (B, 3, H, W)
            targets = targets.to(device)          # shape: (B, H, W), values in [0..19]


            targets = targets.squeeze(1)

            outputs = model(images)

            # If model returns auxiliary outputs as a tuple/list, take the first (main) output
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # unique_values = torch.unique(targets)
            # print("Unique target values:", unique_values)

            # outputs: (B, 20, H, W)
            # targets[targets >= 20] = 255

            optimizer.zero_grad() 

            loss = criterion(outputs, targets)
            loss.backward(); optimizer.step()

            loop.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
        print(f"Epoch {epoch+epochnum}/{epochnum+9}, Loss: {epoch_loss/len(train_loader):.4f}")
        torch.save({
          'epoch': epoch + epochnum,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict()
      }, f'./checkpoints/Checkpoint{epochnum + epoch}.pth')
        scheduler.step()

    # Save checkpoint

if __name__ == '__main__':
    train()
