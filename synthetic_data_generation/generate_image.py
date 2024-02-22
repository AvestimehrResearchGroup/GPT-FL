import os
import sys
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../syn_dataset")))

def cifar_10_generate(device_num, gpu_number, train_number):
    label = ['an Airplane', 'an Automobile', 'a Bird', 'a Cat', 'a Deer', 'a Dog', 'a Frog', 'a Horse', 'a Ship', 'a Truck']
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda:" + str(device_num) if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    text_path = "syn_dataset/cifar10_promptlist.pkl"
    with open(text_path, 'rb') as f:
        prompt_dict = pickle.load(f)
    
    train_image_number = math.ceil(train_number / gpu_number / len(label))
    # test_image_number = math.ceil(test_number / gpu_number / len(label))
    for i in label:
        class_name = i.replace(' ','_')
        for j in tqdm(range(train_image_number)):
            prompt_id = np.random.randint(low=0, high=200)
            prompt = prompt_dict[i][prompt_id]
            ugs = np.random.randint(low=1, high=6)
            
            save_path = args.save_path + '/' + class_name + '_train_' + str(device_num) + '_' + str(j) + '.png'
            if Path(save_path).exists(): 
                print(f'skipping file {save_path}')
                continue
            image = pipe(prompt, guidance_scale=ugs).images[0]
            
            save_file_path = args.save_path + '/' + class_name + '_train_' + str(device_num) + '_' + str(j) + '.png'
            image.save(save_file_path)

def flower_102_generate(device_num, gpu_number, train_number):
    label = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda:" + str(device_num) if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    train_image_number = math.ceil(train_number / gpu_number / len(label))
    # test_image_number = math.ceil(test_number / gpu_number / len(label))
    for i in label:
        class_name = i.replace(' ','_')
        for j in tqdm(range(train_image_number)):
            # prompt_id = np.random.randint(low=0, high=200)
            prompt = "a flower photo of " + i
            ugs = np.random.randint(low=1, high=6)
            
            save_path = args.save_path + '/' + class_name + '_train_photo_add_' + str(device_num) + '_' + str(j) + '.png'
            if Path(save_path).exists(): 
                print(f'skipping file {save_path}')
                continue
            image = pipe(prompt, guidance_scale=ugs).images[0]
            
            save_file_path = args.save_path + '/' + class_name + '_train_photo_add_' + str(device_num) + '_' + str(j) + '.png'
            image.save(save_file_path)
        
def food_101_generate(
    device_num, 
    gpu_number, 
    train_number
):
    # Available labels
    label = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
    
    # Choose models
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda:" + str(device_num) if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    train_image_number = math.ceil(train_number / gpu_number / len(label))
    
    # Iterate over labels
    for i in label:
        class_name = i.replace(' ','_')
        for j in tqdm(range(train_image_number)):
            save_file_path = Path(args.save_path).joinpath(f"{class_name}_train_photo_add_{device_num}_{j}.png")
            # Skip the Generation if performed, as each generation takes 5-10 secs
            if Path(save_file_path).exists(): 
                print(f'skipping file {save_file_path}')
                continue
            # Class prompt case
            if args.generate_method == "class_prompt":
                prompt = "a food photo of " + i
                image = pipe(prompt).images[0]
            elif args.generate_method == "multi_domain":
                domain_list = ["photo", "drawing", "painting", "sketch", "collage", "poster", "digital art image", "rock drawing", "stick figure", "3D rendering"]
                domain_idx = np.random.randint(low=0, high=len(domain_list)-1)
                domain = domain_list[domain_idx]
                prompt = f"a food {domain} of " + i
                image = pipe(prompt).images[0]
            elif args.generate_method == "ucg":
                prompt = "a food photo of " + i
                ucg = np.random.randint(low=1, high=6)
                image = pipe(prompt, guidance_scale=ucg).images[0]
            image.save(save_file_path)

def covid_generate(device_num, gpu_number):
    label = ['NORMAL', 'Viral Pneumonia', 'COVID']
    prompt = ['a chest X-ray image of a healthy human lung, with no signs of infection, disease, or abnormality, for control comparison', 
    'a high-resolution chest X-ray image shows the characteristic signs of viral pneumonia.', 'a high-resolution chest X-ray image showing typical features of COVID-19']
    image_ratio = [0.6, 0.2, 0.2]
    train_number = 10192 + 1345 + 3616
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda:" + str(device_num) if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    train_image_number = math.ceil(train_number / gpu_number)
    normal_number = round(train_image_number * image_ratio[0])
    vp_number = round(train_image_number * image_ratio[1])
    covid_number = round(train_image_number * image_ratio[2])

    for idx in tqdm(range(train_image_number)):
        if idx <= normal_number:
            given_prompt = prompt[0]
            given_label = label[0]
        elif idx > normal_number and idx <= normal_number + vp_number:
            given_prompt = prompt[1]
            given_label = label[1]
        else:
            given_prompt = prompt[2]
            given_label = label[2]
        ugs = np.random.randint(low=1, high=6)
        save_path = args.save_path + '/' + given_label + '_train_photo_' + str(device_num) + '_' + str(idx) + '.png'
        image = pipe(given_prompt, guidance_scale=ugs).images[0]
        image.save(save_path)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Image generation')
    parser.add_argument(
        '--gpu_number', 
        default=3,
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
        default="syn_dataset",
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
        default="food_101",
        type=str, 
        help='dataset: cifar10, cifar100, food_101, flower_102'
    )
    
    args = parser.parse_args()
    
    Path.mkdir(
        Path(args.save_path, args.dataset, args.generate_method), 
        parents=True, 
        exist_ok=True
    )
    args.save_path = args.save_path + "/" + args.dataset + "/" + args.generate_method
    
    train_number = 75750
    test_number = 0
    prompt_number = 0
    gpu_number = args.gpu_number
    device_num = args.device_num
    if args.dataset == "flower_102":
        flower_102_generate(
            device_num, 
            gpu_number, 
            train_number
        )
    if args.dataset == "cifar10":
        cifar_10_generate(
            device_num, 
            gpu_number, 
            train_number, 
        )
    elif args.dataset == "food_101":
        food_101_generate(
            device_num, 
            gpu_number, 
            train_number
        )

