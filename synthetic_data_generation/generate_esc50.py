import os
import sys
import math
import torch
import scipy
import pickle
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from diffusers import AudioLDMPipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../syn_dataset")))


def esc50_generate(
    device_num, 
    gpu_number, 
    train_number
):
    # Available labels
    # Fetch all audio files
    meta_df = pd.read_csv(str(Path(args.raw_data_dir).joinpath('meta/esc50.csv')))
    label_list = list(meta_df["category"].unique())
    
    # Choose models
    model_id = "cvssp/audioldm"
    pipe = AudioLDMPipeline.from_pretrained(
        model_id,
        cache_dir="/media/data/projects/speech-privacy/fl-syn/fl-syn/syn_dataset/"
    )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    train_audio_number = math.ceil(train_number / gpu_number / len(label_list))
    
    # Iterate over labels
    for label in label_list:
        class_name = label.replace('_',' ')
        for j in tqdm(range(train_audio_number)):
            save_file_path = Path(args.save_path).joinpath(args.dataset, args.generate_method, f"{label}_add_{device_num}_{j}.wav")
            # Skip the Generation if performed, as each generation takes 5-10 secs
            if Path(save_file_path).exists(): 
                print(f'skipping file {save_file_path}')
                continue
            # Class prompt case
            if args.generate_method == "class_prompt":
                prompt = f"a sound of {class_name}"
                audio = pipe(
                    prompt, 
                    num_inference_steps=100, 
                    audio_length_in_s=5.0
                ).audios[0]

            elif args.generate_method == "multi_domain":
                domain_list = ["in a studio", "in the field", "at home", "in an office"]
                domain_idx = np.random.randint(low=0, high=len(domain_list)-1)
                domain = domain_list[domain_idx]
                prompt = f"{class_name} {domain}"
                audio = pipe(
                    prompt, 
                    num_inference_steps=100, 
                    audio_length_in_s=5.0
                ).audios[0]

            elif args.generate_method == "ucg":
                prompt = f"{class_name}"
                ucg = np.random.randint(low=1, high=6)
                audio = pipe(
                    prompt, 
                    num_inference_steps=100, 
                    audio_length_in_s=5.0, 
                    guidance_scale=ucg
                ).audios[0]
                
            torchaudio.save(
                str(save_file_path), 
                torch.tensor(audio).unsqueeze(dim=0), 
                16000
            )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Audio generation')
    parser.add_argument(
        '--gpu_number', 
        default=1,
        type=int, 
        help='Total number of gpus'
    )
    
    parser.add_argument(
        '--device_num', 
        default=1,
        type=int, 
        help='device number'
    )
    
    parser.add_argument(
        '--save_path', 
        default="../syn_dataset",
        type=str, 
        help='Save folder path'
    )
    
    parser.add_argument(
        '--generate_method', 
        default="class_prompt",
        type=str, 
        help='generate method: class_prompt, multi_domain, ucg'
    )
    
    parser.add_argument(
        '--dataset', 
        default="esc50",
        type=str, 
        help='dataset: esc50, speech_commands, slurp, music'
    )

    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="../ESC-50-master",
        help="Raw data path of ESC-50 dataset",
    )
    
    args = parser.parse_args()
    
    Path.mkdir(
        Path(args.save_path, args.dataset, args.generate_method), 
        parents=True, 
        exist_ok=True
    )
    
    gpu_number = args.gpu_number
    device_num = args.device_num
    
    train_number = 2000
    esc50_generate(
        device_num, 
        gpu_number, 
        train_number
    )
    