import os

import torch
import time
from torch import nn
from utils import setup_seed, get_fr_model, initialize_model, asr_calculation,get_train_model
import os
from FaceParsing.interface import FaceParsing
from dataset import base_dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from torch.utils.data import Subset
from PIL import Image


seed = 0
setup_seed(0)

@torch.no_grad()
def main(args):
    h = 512
    w = 512
    txt = ''
    ddim_steps = 50
    scale = 0
    classifier_scale = args.s
    batch_size = 1
    num_workers = 0
    
    config = "config/attack_faster_rcnn_50.yaml"
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor()])
    if args.dataset == 'coco':
        dataset = base_dataset(dir='./coco', transform=transform)
    elif args.dataset == 'voc':
        dataset = base_dataset(dir='./voc', transform=transform)
    # dataset = base_dataset(dir='./s', transform=transform)
    dataset = Subset(dataset, [x for x in range(args.num)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml', 
                               '512-inpainting-ema.ckpt')
    model = sampler.model


    attack_model_names = [args.model]
    attack_model_dict = {'faster_rcnn':[get_fr_model('faster_rcnn'),get_train_model('faster_rcnn')],'yolo':[get_fr_model('yolo'),get_train_model('yolo')]}
    cos_sim_scores_dict = {args.model: []}
    
    json_file = 'instances_val2017.json'
    
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        classifier = {k: v for k, v in attack_model_dict.items() if k == attack_model_name}
        with torch.no_grad():
            for i, (image, tgt_image,image_init) in enumerate(dataloader):
                tgt_image = tgt_image.to(device)         
                B = image.shape[0]  
                image_init = image_init.numpy().squeeze()
                mask = torch.ones((B, 1, h, w), dtype=torch.bool)
                mask = (mask == 0).float().cpu()
                masked_image = image * (mask <= 0.5)
 
                batch = {
                    "image": image.to(device),
                    "txt": batch_size * [txt],
                    "mask": mask.to(device),
                    "masked_image": masked_image.to(device),
                }

                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [batch_size, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(batch_size, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]
                
                # start code
                _t = args.t  # 0-999
                z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                t = torch.tensor([_t] * batch_size, device=device)
                z_t = model.q_sample(x_start=z, t=t)
                img_t = model.decode_first_stage(z_t)
                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    batch_size,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=z_t,
                    _t=_t + 1,
                    log_every_t=1,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    x_target=tgt_image,
                    x_init=image_init
                )

                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='faster_rcnn')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--num', type=int, default='4')
    parser.add_argument('--t', type=int, default=999)
    parser.add_argument('--save', type=str, default='res')
    parser.add_argument('--s', type=int, default=10)
    args = parser.parse_args()
    
    main(args)
