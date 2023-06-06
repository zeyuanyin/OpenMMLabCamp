import os
import random
import shutil

def split(class_name, raw_data_dir):
    # Create train, validation, and test directories
    train_dir = os.path.join(raw_data_dir, 'train',class_name)
    val_dir = os.path.join(raw_data_dir, 'val', class_name)
    # test_dir = os.path.join(raw_data_dir, 'test', class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(test_dir, exist_ok=True)

    # Get all files in the dataset directory
    class_path = os.path.join(raw_data_dir, class_name)
    files = os.listdir(class_path)
    random.shuffle(files)

    # Calculate split indices
    # train_ratio = 0.7
    # val_ratio = 0.15
    # test_ratio = 0.15
    train_ratio = 0.8
    val_ratio = 0.2
    train_index = int(len(files) * train_ratio)
    val_index = int(len(files) * (train_ratio + val_ratio))

    # Move files to train directory
    for file in files[:train_index]:
        src = os.path.join(class_path, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)

    # Move files to validation directory
    for file in files[train_index:val_index]:
        src = os.path.join(class_path, file)
        dst = os.path.join(val_dir, file)
        shutil.copy(src, dst)

    # Move files to test directory
    # for file in files[val_index:]:
    #     src = os.path.join(class_path, file)
    #     dst = os.path.join(test_dir, file)
    #     shutil.copy(src, dst)

if __name__ == '__main__':
    # Define dataset directory and split ratios
    raw_data_dir = '/home/zeyuan.yin/OpenMMLabCamp/homework-2/data'

    from tqdm import tqdm
    for class_name in tqdm(os.listdir(raw_data_dir)):
        # print(class_path)
        split(class_name, raw_data_dir)



