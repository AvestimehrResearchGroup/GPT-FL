import os
import sys
import math
import torch
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../syn_dataset")))

def gcommand_generate(
    device_num, 
    gpu_number, 
    train_number
):
    # Available labels
    label_list = [
        "backward",
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "follow",
        "forward",
        "four",
        "go",
        "happy",
        "house",
        "learn",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "visual",
        "wow",
        "yes",
        "zero"
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Choose models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", 
        split="validation",
    )
    train_audio_number = math.ceil(train_number / gpu_number / len(label_list))
    
    # Iterate over labels
    for label in label_list:
        class_name = label.replace(' ','_')
        for j in tqdm(range(train_audio_number)):
            save_file_path = Path(args.save_path).joinpath(args.dataset, args.generate_method, f"{class_name}_add_{device_num}_{j}.wav")
            # Skip the Generation if performed, as each generation takes 5-10 secs
            if Path(save_file_path).exists(): 
                print(f'skipping file {save_file_path}')
                continue
            
            # Class prompt case
            if args.generate_method == "class_prompt":
                inputs = processor(
                    text=f"{class_name}", return_tensors="pt"
                ).to(device)
                speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to(device)
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder).cpu().detach()

            elif args.generate_method == "multi_speaker":
                speaker_idx = np.random.randint(low=0, high=len(embeddings_dataset)-1)
                speaker_embeddings = torch.tensor(embeddings_dataset[speaker_idx]["xvector"]).unsqueeze(0).to(device)
    
                inputs = processor(
                    text=f"{class_name}", return_tensors="pt"
                ).to(device)
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder).cpu().detach()

            torchaudio.save(
                str(save_file_path), 
                torch.tensor(speech).unsqueeze(dim=0), 
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
        help='generate method: class_prompt, multi_speaker, ucg'
    )
    
    parser.add_argument(
        '--dataset', 
        default="speech_commands",
        type=str, 
        help='dataset: esc50, speech_commands, slurp, music'
    )
    
    args = parser.parse_args()
    
    Path.mkdir(
        Path(args.save_path, args.dataset, args.generate_method), 
        parents=True, 
        exist_ok=True
    )
    
    gpu_number = args.gpu_number
    device_num = args.device_num
    train_number = 60000
    gcommand_generate(
        device_num, 
        gpu_number, 
        train_number
    )
