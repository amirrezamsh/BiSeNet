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
weights_path = './checkpoints/bisenetv2_finetuned_epoch10.pth'
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


    param_num = 0
    for name, param in model.named_parameters():
      if (
          'conv_out' in name               # all 1×1 classifier heads
          or name.startswith('head.conv.') # the head fusion conv
          or name.startswith('bga.')       # all of the BGA module
      ):
          param.requires_grad = True
          param_num += 1
      else:
          param.requires_grad = False

    print(f"✅Gradiation turned on for {param_num} parameters")

    criterion = nn.CrossEntropyLoss()

    def compute_loss(outs, targets):
      main, a2, a3, a4, a5 = outs
      loss_main = criterion(main, targets)
      loss_aux  = 0.4*(criterion(a2,targets)
                    +criterion(a3,targets)
                    +criterion(a4,targets)
                    +criterion(a5,targets))
      return loss_main + loss_aux
    
    # 2) Optimizer
    params = [
      {'params': [p for n,p in model.named_parameters() if 'conv_out' in n], 'lr':1e-4},
      {'params': [p for n,p in model.named_parameters() if 'head.conv.' in n or 'bga.' in n], 'lr':5e-5},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    model.train()

    for epoch in range(20):
      epoch_loss = 0.0
      loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
      for images, targets in loop:
          images = images.to(device)            # shape: (B, 3, H, W)
          targets = targets.to(device)          # shape: (B, H, W), values in [0..19]


          targets = targets.squeeze(1)


          outputs = model(images)


          # unique_values = torch.unique(targets)
          # print("Unique target values:", unique_values)

          # outputs: (B, 20, H, W)
          # targets[targets >= 20] = 255

          optimizer.zero_grad()

          loss = compute_loss(outputs, targets)
          loss.backward(); optimizer.step()

          loop.set_postfix(loss=loss.item())
          epoch_loss += loss.item()
      scheduler.step()
      print(f"Epoch {epoch+epochnum}/{epochnum+19}, Loss: {epoch_loss/len(train_loader):.4f}")
      torch.save(model.state_dict(), f"./checkpoints/bisenetv2_finetuned2_epoch{epoch + epochnum}.pth")

    torch.save(model.state_dict(), "./checkpoints/bisenetv2_finetuned2.pth")

if __name__ == '__main__':
    train()









