import torch
import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import utils

def run(input_path, output_path, model_type:str,align_rel_imgs=False, *args, **kwargs):
    """"""

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # select model
    model = torch.hub.load("intel-isl/MiDaS",model_type)

    model.eval()
    model.to(device)

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ['DPT_Large', 'DPT_Hybrid', 'MiDaS']:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    
    # get RGB input
    rgb_names = glob(f"{input_path}/*rgb.png")
    num_images = len(rgb_names)
    print(f"No. of images: {num_images}")
    print(f"Output path: {output_path}")
    print(f"Model Name: {model_type}")

    print("start processing...")

    

    for i, img_name in enumerate(rgb_names):
        print(f" processing {img_name} ({i+1}/{num_images})")

        # input 

        img = np.array(Image.open(img_name))

        sample = transform(img).to(device)        
        with torch.no_grad():
            
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            # To be used for absolute depth estimation
            if align_rel_imgs:
                prediction = utils.align_depth(img[:,:,0], prediction)
        # save output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction, bits=2)
        print(f"{filename} written")

    print(f"model inference finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument("-t", "--model_type", 
        default='DPT_Hybrid',
        help="model types: DPT_Large, DPT_Hybrid, MiDaS, MiDaS_small")

    parser.add_argument('-a','--align_rel_imgs', 
        default=False, 
        type=lambda x: (str(x).lower() == 'true'), 
        help="""Aligns images from relative to absolute depth. \n
        Refer issue: https://github.com/isl-org/MiDaS/issues/171"""
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    run(args.input_path, args.output_path, args.model_type, args.align_rel_imgs)
