import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset


def create_mask(mask_creator_dir, class_dir, mask_dir):

    pallete_dict = {'Background': [], 'Face': [], 'Upper-clothes':[], 'Pants':[], 'Arm':[]}



    val_list = [[0],[1,2,4,13],[5,6,7,11],[8,9,10,12,18,19],[0,1,2,4,13,14,15,16,17]]
    for c,j in enumerate(pallete_dict.keys()):
    val = val_list[c]
    pallete = []
    for i in range(60):
        
        if len(val) == 1:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)):
            pallete.append(0)
        else:
            pallete.append(255)
        if len(val) == 2:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)):
            pallete.append(0)
        else:
            pallete.append(255)
        if len(val) == 3:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)) or (i >= (val[2]*3)) & (i < ((val[2]+1)*3)):
            pallete.append(255)
        else:
            pallete.append(0)
        if len(val) == 4:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)) or (i >= (val[2]*3)) & (i < ((val[2]+1)*3)) or (i >= (val[3]*3)) & (i < ((val[3]+1)*3)):
            if j == 'Face':
                pallete.append(255)
            else:
                pallete.append(0)
        else:
            if j == 'Face':
                pallete.append(0)
            else:
                pallete.append(255)
        if len(val) == 5:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)) or (i >= (val[2]*3)) & (i < ((val[2]+1)*3)) or (i >= (val[3]*3)) & (i < ((val[3]+1)*3)) or (i >= (val[4]*3)) & (i < ((val[4]+1)*3)):
            pallete.append(0)
        else:
            pallete.append(255)
        if len(val) == 6:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)) or (i >= (val[2]*3)) & (i < ((val[2]+1)*3)) or (i >= (val[3]*3)) & (i < ((val[3]+1)*3)) or (i >= (val[4]*3)) & (i < ((val[4]+1)*3)) or (i >= (val[5]*3)) & (i < ((val[5]+1)*3)):
            pallete.append(0)
        else:
            pallete.append(255)
        if len(val) == 9:
        if (i >= (val[0]*3)) & (i < ((val[0]+1)*3)) or (i >= (val[1]*3)) & (i < ((val[1]+1)*3)) or (i >= (val[2]*3)) & (i < ((val[2]+1)*3)) or (i >= (val[3]*3)) & (i < ((val[3]+1)*3)) or (i >= (val[4]*3)) & (i < ((val[4]+1)*3)) or (i >= (val[5]*3)) & (i < ((val[5]+1)*3)) or (i >= (val[6]*3)) & (i < ((val[6]+1)*3)) or (i >= (val[7]*3)) & (i < ((val[7]+1)*3)) or (i >= (val[8]*3)) & (i < ((val[8]+1)*3)):
            pallete.append(255)
        else:
            pallete.append(0)

    pallete_dict[j] = pallete


    dataset_settings = {
        'lip': {
            'input_size': [473, 473],
            'num_classes': 20,
            'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                    'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                    'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        },
        'atr': {
            'input_size': [512, 512],
            'num_classes': 18,
            'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                    'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
        },
        'pascal': {
            'input_size': [512, 512],
            'num_classes': 7,
            'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
        }
    }



    gpus = [0]
    assert len(gpus) == 1
    if not gpus == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    num_classes = dataset_settings['lip']['num_classes']
    input_size = dataset_settings['lip']['input_size']
    label = dataset_settings['lip']['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(os.path.join(mask_creator_dir,'checkpoints/final.pth'))['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=class_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists('outputs_training'):
        os.makedirs('outputs_training')

    folder_path = []
    for folder in pallete_dict.keys():
        folder_path.append(os.path.join(mask_dir, folder))
    if not os.path.exists(os.path.join(mask_dir, folder)):
        os.makedirs(os.path.join(mask_dir, folder))


    # palette = get_palette(num_classes)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))

            for loc, key in  enumerate(pallete_dict.keys()):
            parsing_result_path = os.path.join(folder_path[loc], img_name)

            output_img.putpalette(pallete_dict[key])
            output_img.save(parsing_result_path)

