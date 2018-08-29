import os
import numpy as np
import random as rand
from PIL import Image

import sys
sys.path.append("../")
import const

IGNORE_FILE = [".DS_Store", ".gitignore", ".gitkeep"]


class DataOpt():
    @classmethod
    def __ignore_file(cls, file_list, remove_files):
        for remove_file in remove_files:
            while (remove_file in file_list):
                file_list.remove(remove_file)
        return file_list[::]

    @classmethod
    def img_open(cls, filepath):
        img = Image.open(filepath)
        img = img.convert("RGB")
        img = img.resize((const.IMG_SIZE, const.IMG_SIZE))
        img = np.array(img)
        img = (img / 255.)
        return np.copy(img)

    @classmethod
    def make_train_data_all(cls, data_file_path):
        image_list = []
        img_files = DataOpt.__ignore_file(
            os.listdir(data_file_path), IGNORE_FILE)

        for img_file in img_files:
            filepath = os.path.join(data_file_path, img_file)
            img = DataOpt.img_open(filepath)
            image_list.append(img)

        image_list = np.array(image_list)
        print(image_list.shape)
        return image_list, image_list

    @classmethod
    def make_train_data_random_choice(cls, data_file_path, data_size):
        image_list = []
        files = DataOpt.__ignore_file(
            os.listdir(data_file_path), IGNORE_FILE)

        while(True):
            idx = rand.randint(0, len(files) - 1)
            if DataOpt.__is_img(files[idx]):
                filepath = os.path.join(data_file_path, files[idx])
                img = DataOpt.img_open(filepath)
                image_list.append(img)
            if len(image_list) >= data_size:
                break

        image_list = np.array(image_list)
        print(image_list.shape)
        return image_list, image_list

    @classmethod
    def make_train_data_random_choice_two_img_set(cls, data_file_path, data_size):
        image_list = []
        files = DataOpt.__ignore_file(
            os.listdir(data_file_path), IGNORE_FILE)

        while(True):
            idx1 = rand.randint(0, len(files) - 1)
            idx2 = rand.randint(0, len(files) - 1)
            if DataOpt.__is_img(files[idx1]) and DataOpt.__is_img(files[idx2]):
                filepath1 = os.path.join(data_file_path, files[idx1])
                img1 = DataOpt.img_open(filepath1)
                filepath2 = os.path.join(data_file_path, files[idx2])
                img2 = DataOpt.img_open(filepath2)
                image_list.append([img1, img2])
            if len(image_list) >= data_size:
                break

        image_list = np.array(image_list)
        print(image_list.shape)
        return image_list, image_list

    @classmethod
    def __is_img(cls, file_name):
        return (DataOpt.__get_ext(file_name) == '.png') or \
            (DataOpt.__get_ext(file_name) == '.jpg') or \
            (DataOpt.__get_ext(file_name) == '.jpeg')

    @classmethod
    def __get_ext(cls, file_name):
        _, ext = os.path.splitext(file_name)
        return ext

    @classmethod
    def load_data_set(cls, load_train_file_name):
        train_data = np.load(load_train_file_name)['train_data']
        print("train shape", train_data.shape)
        teach_data = np.load(load_train_file_name)['teach_data']
        print("teach shape", teach_data.shape)
        return train_data, teach_data

    @classmethod
    def save_data_set(cls, train_data, teach_data, save_train_file_name):
        np.savez_compressed(save_train_file_name,
                            train_data=train_data, teach_data=teach_data)
        print("save ", save_train_file_name)
