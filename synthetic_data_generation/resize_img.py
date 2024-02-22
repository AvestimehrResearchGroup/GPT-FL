from PIL import Image
import os
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image resize')
    parser.add_argument(
        '--original_image_path', 
        # default="/project/shrikann_35/tiantiaf/fl-syn/syn_dataset/cifar10/train_dataset",
        default="/scratch1/tiantiaf/fl-syn/syn_dataset/cifar10/train_dataset/",
        type=str, 
        help='Original generated image path'
    )

    parser.add_argument(
        '--resize_image_path', 
        default="/scratch1/tiantiaf/fl-syn/syn_dataset/cifar10/train_resize_dataset_20",
        type=str, 
        help='Resized image path'
    )
    args = parser.parse_args()

    # Set the directory containing the images
    directory = args.original_image_path

    # Set the output directory for the resized images
    output_directory = args.resize_image_path

    # Set the new size for the images
    new_size = (32, 32)

    # Loop over all the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            device = int(filename.split("train_")[1].split("_")[0])
            if device >= 30: continue
            if os.path.exists(os.path.join(output_directory, filename)):
                continue
            # Open the image
            with Image.open(os.path.join(directory, filename)) as img:
                # import pdb
                # pdb.set_trace()
                # Resize the image
                resized_img = img.resize(new_size)
                # Save the resized image
                resized_img.save(os.path.join(output_directory, filename))
