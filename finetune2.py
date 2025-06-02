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
weights_path = './checkpoints/Checkpoint10.pth'
# cityscapes_root = r"D:\semester_3\AML\project\datasets\cityscapes"
cityscapes_root = '../cityscapes'

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
    # -1: 0
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




def train() :

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
    
    
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage, weights_only=False)

    epochnum = checkpoint['epoch'] + 1

    model = load_my_state_dict(model, checkpoint['model_state_dict'])
    
    model.to(device)

    print("✅Model weights updated sucssessfully")


    param_num = 0
    for name, param in model.named_parameters():
        if 'conv_out' in name \
        or name.startswith('head.conv') and 'conv_out' not in name \
        or name.startswith('segment.S5_4') \
        or name.startswith('segment.S5_5') \
        or name.startswith('bga'):
            param.requires_grad = True
            param_num+=1
        else :
            param.requires_grad = False

    print(f"✅Gradiation turned on for {param_num} parameters")

    criterion = nn.CrossEntropyLoss()

    # 
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'conv_out' in name or name.startswith('head.conv'):
                head_params.append(p)
            else:
                backbone_params.append(p)

    optimizer = optim.Adam([
        {'params': head_params,     'lr': 1e-4},
        {'params': backbone_params, 'lr': 1e-5},
    ], weight_decay=1e-4)

    start_epoch = checkpoint.get('epoch', 0) 

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    scheduler.last_epoch = start_epoch
    # 

    def compute_loss(outs, targets):
      main, a2, a3, a4, a5 = outs
      loss_main = criterion(main, targets)
      loss_aux  = 0.4*(criterion(a2,targets)
                    +criterion(a3,targets)
                    +criterion(a4,targets)
                    +criterion(a5,targets))
      return loss_main + loss_aux
    
    # 2) Optimizer

    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    model.train()

    for epoch in range(10):
      epoch_loss = 0.0
      loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
      for images, targets in loop:
          images = images.to(device)            # shape: (B, 3, H, W)
          targets = targets.to(device)          # shape: (B, H, W), values in [0..19]

          #targets = targets.squeeze(1)

          outputs = model(images)

          optimizer.zero_grad()

          loss = compute_loss(outputs, targets)
          loss.backward(); optimizer.step()

          loop.set_postfix(loss=loss.item())
          epoch_loss += loss.item()
      scheduler.step()
      print(f"Epoch {epoch+epochnum}/{epochnum+9}, Loss: {epoch_loss/len(train_loader):.4f}")
      # Save full training state
      torch.save({
          'epoch': epoch + epochnum,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict()
      }, f'./checkpoints/Checkpoint{epochnum + epoch}.pth')


    #torch.save(model.state_dict(), "./checkpoints/bisenetv2_finetuned2.pth")

if __name__ == '__main__':
    train()








