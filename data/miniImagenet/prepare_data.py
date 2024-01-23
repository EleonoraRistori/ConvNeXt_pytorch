import os
import csv
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def save_images_to_directory(images, labels_dict, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Save images to the directory
    for label, indices in labels_dict.items():
        label_directory = os.path.join(output_directory)
        os.makedirs(label_directory, exist_ok=True)

        for idx in indices:
            image_filename = f"{label}_{idx}.png"
            image_path = os.path.join(label_directory, image_filename)
            cv2.imwrite(image_path, (images[idx] * 255).astype(np.uint8))

def create_csv_file(labels_dict, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image_Path', 'Label'])

        for label, indices in labels_dict.items():
            for idx in indices:
                image_filename = f"{label}_{idx}.png"
                csv_writer.writerow([image_filename, label])

def split_dataset(csv_file, images_directory, train_size=None, val_size=None, test_size=None, random_seed=42):
    # Read CSV file to get image paths and labels
    image_paths = []
    labels = []
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            image_paths.append(row[0])  # Assuming image paths are in the first column
            labels.append(row[1])  # Assuming labels are in the second column

    # Split paths and labels into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=val_size + test_size, random_state=random_seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=test_size, random_state=random_seed
    )


    # Create CSV files for train, validation, and test sets
    create_split_file(train_paths, train_labels, 'train.csv')
    create_split_file(val_paths, val_labels, 'val.csv')
    create_split_file(test_paths, test_labels, 'test.csv')



def create_split_file(image_paths, labels, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image_Path', 'Label'])
        for image_path, label in zip(image_paths, labels):
            csv_writer.writerow([image_path, label])


if __name__ == "__main__":
    # Load your dataset from the pickle file
    with open('mini-imagenet-cache-train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('mini-imagenet-cache-val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('mini-imagenet-cache-test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    merged_labels = {}
    merged_labels.update(train_data['class_dict'])
    merged_labels.update(test_data['class_dict'])
    merged_labels.update(val_data['class_dict'])
    merged_data = {
        'data': np.concatenate((train_data['image_data'], val_data['image_data'], test_data['image_data']), axis=0),
        'labels': merged_labels
    }


    # Specify the output directory and CSV file
    output_directory = 'images'
    output_csv_file = 'labels.csv'
    # Save images to the directory
    save_images_to_directory(merged_data['data'], merged_data['labels'], output_directory)
    create_csv_file(merged_data['labels'], output_csv_file)

    # Specify the exact number of images for each set
    train_size = 45000
    val_size = 5000
    test_size = 10000

    # Split the dataset and create CSV files
    split_dataset(output_csv_file, output_directory, train_size, val_size, test_size)



