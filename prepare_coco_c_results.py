import os
import csv
import re

def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        iou_5095 = None
        iou_50 = None
        for line in lines:
            # Extract the line with IoU=0.50:0.95 where area=all
            if "AP) @[" in line and "IoU=0.50:0.95" in line and "| area=   all |" in line:
                iou_5095 = float(line.split('=')[-1].strip())
            # Extract the line with IoU=0.50 where area=all
            elif "AP) @[" in line and "IoU=0.50 " in line and "| area=   all |" in line:
                iou_50 = float(line.split('=')[-1].strip())
    return iou_5095, iou_50

def process_model_directory(input_directory, output_directory):
    # Regex pattern to extract corruption and severity level
    pattern = re.compile(r'([a-zA-Z_]+)_coco17_([a-zA-Z_]+)(\d)_results\.txt')

    # Loop through each model directory
    for model_folder in os.listdir(input_directory):
        model_path = os.path.join(input_directory, model_folder)
        if os.path.isdir(model_path):
            data = {}
            print(f"Processing files in {model_path}...")

            # Process each file in the model's directory
            for filename in os.listdir(model_path):
                match = pattern.match(filename)
                if match:
                    corruption = match.group(2)
                    severity = match.group(3)
                    file_path = os.path.join(model_path, filename)
                    iou_5095, iou_50 = extract_data_from_file(file_path)

                    # Populate the data dictionary
                    if iou_5095 is not None and iou_50 is not None:
                        if corruption not in data:
                            data[corruption] = {}
                        data[corruption][severity] = (iou_5095, iou_50)

            # Write the results to a CSV file
            output_file_path = os.path.join(output_directory, f"{model_folder}_results.csv")
            with open(output_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Corruption", "Severity", "IoU=0.50:0.95", "IoU=0.50"])
                
                for corruption, severities in sorted(data.items()):
                    for severity, metrics in sorted(severities.items(), key=lambda item: int(item[0])):
                        csvwriter.writerow([corruption, severity] + list(metrics))

# Example usage
input_dir = '/home/opendict_detection/final_results'  # Path to the directory containing all model folders
output_dir = '/home/opendict_detection/prepared_results'  # Path where the CSV files will be saved
process_model_directory(input_dir, output_dir)
