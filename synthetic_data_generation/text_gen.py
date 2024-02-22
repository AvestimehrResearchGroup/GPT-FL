import sys, random
from keytotext import pipeline
import pickle

# labels = \
# ['a Annual Crop Land', 'a Forest', 'a Herbaceous Vegetation Land', 'a Highway or Road', 'a Industrial Building', 'a Pasture Land', 'a Permanent Crop Land', 'a Residential Building', 'a River', 'a Sea or Lake']


nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")


def word2sentence(classnames, num=200, save_path=""):
    sentence_dict = {}
    for n in classnames:
        sentence_dict[n] = []
    for n in classnames:
        for i in range(num + 50):
            sentence = nlp([n], num_return_sequences=1, do_sample=True)
            sentence_dict[n].append(sentence)
            print(sentence)

    # remove duplicate
    sampled_dict = {}
    for k, v in sentence_dict.items():
        v_unique = list(set(v))
        sampled_v = random.sample(v_unique, num)
        sampled_dict[k] = sampled_v

    r = open(save_path, "wb")
    pickle.dump(sampled_dict, r)
    r.close()


if __name__ == "__main__":
    if sys.argv[1] == "cifar10":
        labels = [
            "an Airplane",
            "an Automobile",
            "a Bird",
            "a Cat",
            "a Deer",
            "a Dog",
            "a Frog",
            "a Horse",
            "a Ship",
            "a Truck",
        ]
        save_path = 'syn_dataset/cifar10/promptlist.pkl'
    if sys.argv[1] == "caltech101":
        labels = [
            "trilobite",
            "face",
            "pagoda",
            "tick",
            "inlineskate",
            "metronome",
            "accordion",
            "yinyang",
            "soccerball",
            "spotted cat",
            "nautilus",
            "grand-piano",
            "crayfish",
            "headphone",
            "hawksbill",
            "ferry",
            "cougar-face",
            "bass",
            "ketch",
            "lobster",
            "pyramid",
            "rooster",
            "laptop",
            "waterlilly",
            "wrench",
            "strawberry",
            "starfish",
            "ceilingfan",
            "seahorse",
            "stapler",
            "stop-sign",
            "zebra",
            "brontosaurus",
            "emu",
            "snoopy",
            "okapi",
            "schooner",
            "binocular",
            "motorbike",
            "hedgehog",
            "garfield",
            "airplane",
            "umbrella",
            "panda",
            "crocodile-head",
            "llama",
            "windsor-chair",
            "car-side",
            "pizza",
            "minaret",
            "dollarbill",
            "gerenuk",
            "sunflower",
            "rhino",
            "cougar-body",
            "crab",
            "ibis",
            "helicopter",
            "dalmatian",
            "scorpion",
            "revolver",
            "beaver",
            "saxophone",
            "kangaroo",
            "euphonium",
            "flamingo",
            "flamingo-head",
            "elephant",
            "cellphone",
            "gramophone",
            "bonsai",
            "lotus",
            "cannon",
            "wheel-chair",
            "dolphin",
            "stegosaurus",
            "brain",
            "menorah",
            "chandelier",
            "camera",
            "ant",
            "scissors",
            "butterfly",
            "wildcat",
            "crocodile",
            "barrel",
            "joshua-tree",
            "pigeon",
            "watch",
            "dragonfly",
            "mayfly",
            "cup",
            "ewer",
            "octopus",
            "platypus",
            "buddha",
            "chair",
            "anchor",
            "mandolin",
            "electric-guitar",
            "lamp",
        ]
        save_path = 'syn_dataset/caltech101/promptlist.pkl'

    num = sys.argv[2]
    word2sentence(labels, int(num), save_path)

"""
python text_gen.py cifar10 200
"""
