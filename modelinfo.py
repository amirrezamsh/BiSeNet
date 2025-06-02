import sys
import torch

sys.path.append('BiSeNet/lib')
from lib.models.bisenetv2 import BiSeNetV2

weights_path = './checkpoints/checkpoint20.pth'

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

if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint  # raw state_dict

model = load_my_state_dict(model, state_dict)

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
