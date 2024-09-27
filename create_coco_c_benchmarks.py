import os
from PIL import Image
from imagecorruptions import corrupt, get_corruption_names
import numpy as np
# Define the input and base output directory
input_dir = '/home/datasets/coco_17/val2017'
base_output_dir = '/home/datasets/coco_17/'

# Ensure the output base directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Get all image filenames from the input directory
image_filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]


# Loop over each corruption
for severity in range(1, 6):
    # Loop over each severity level
    for corruption in get_corruption_names():
        if corruption not in ['gaussian_noise','impulse_noise', 'shot_noise']:
            # Define the output directory for the current corruption type and severity
            output_dir = os.path.join(base_output_dir, f'val2017_{corruption}{severity}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each image
            for filename in image_filenames:
                # Construct the full file path
                file_path = os.path.join(input_dir, filename)
                
                # Open the image
                with Image.open(file_path) as img:
                    # Convert image to RGB (to ensure compatibility with corrupt function)
                    img = img.convert('RGB')
                    
                    # Apply corruption
                    corrupted_img = corrupt(np.array(img), corruption_name=corruption, severity=severity)
                    
                    # Convert NumPy array back to PIL Image
                    corrupted_img_pil = Image.fromarray(corrupted_img)
                    
                    # Save the corrupted image to the specified directory
                    corrupted_img_pil.save(os.path.join(output_dir, filename))

        print(f'Generated coco validation set of courrption {corruption} and sevirity {severity}')

print("All sets are generated.")
