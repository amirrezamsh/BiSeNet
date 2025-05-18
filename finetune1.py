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


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (update if needed)
weights_path = r"C:\Users\ASUS\Desktop\model_final_v2_city.pth"
cityscapes_root = r"D:\semester_3\AML\project\datasets\cityscapes"

# Ensure output directory exists
os.makedirs("./checkpoints", exist_ok=True)



sys.path.append('BiSeNet/lib')
from lib.models.bisenetv2 import BiSeNetV2

# Define image transformations: resize, to-tensor, normalize
transform = transforms.Compose([
    transforms.Resize((1024, 2048)),  # Cityscapes default resolution (H x W)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Cityscapes uses standard ImageNet normalization:contentReference[oaicite:11]{index=11}
                         std=[0.229, 0.224, 0.225]),
])
# Target transform: convert PIL image to tensor, map 255 -> 19 for background class
def target_transform(pil_img):
    
    label = transforms.functional.pil_to_tensor(pil_img).long()

    # Map unlabeled/void (255) -> class 19 (background)
    label[label == 255] = 19
    return label


def replace_conv_out(layer):
    in_channels = layer[1].in_channels
    layer[1] = nn.Conv2d(in_channels, 20, kernel_size=1)
    nn.init.kaiming_normal_(layer[1].weight, mode='fan_out', nonlinearity='relu')
    if layer[1].bias is not None:
        nn.init.constant_(layer[1].bias, 0)


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
    

    #place this onlt if this is the first time running this file
    # replace_conv_out(model.head.conv_out)
    # replace_conv_out(model.aux2.conv_out)
    # replace_conv_out(model.aux3.conv_out)
    # replace_conv_out(model.aux4.conv_out)
    # replace_conv_out(model.aux5_4.conv_out)


    
    # Load pretrained weights (Cityscapes 19-class model)
    # We load matching layers and skip the final classifier weight mismatch.
    pretrained = torch.load(weights_path, map_location=device)
    model_dict = model.state_dict()
    # Filter out final-layer parameters (shape mismatch) and load the rest
    pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    print("✅Model weights updated sucssessfully")


    for name, param in model.named_parameters():
        if 'conv_out' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


    print("✅Gradiation turned on only on interested layers")

  # Define loss and optimizer (only params with requires_grad=True will be updated)
  
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

  # Training loop (5 epochs)
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
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
            # outputs: (B, 20, H, W)
            targets[targets >= 20] = 255


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

