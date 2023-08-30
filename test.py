import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data
from glob import glob

H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    test_x = sorted(glob(os.path.join("/mnt/batch/tasks/shared/LS_root/mounts/clusters/braincompute/code/Users/raunak.12007198/Splitted_Data_final/Splitted_Data_final/Test", "images", "*.png")))     
    test_y = sorted(glob(os.path.join("/mnt/batch/tasks/shared/LS_root/mounts/clusters/braincompute/code/Users/raunak.12007198/Splitted_Data_final/Splitted_Data_final/Test", "mask_img_labelstest", "*.png")))

    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Evaluation and Prediction """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Create a folder for the current prediction """
        output_folder = os.path.join("results", f"{name}_prediction")
        os.makedirs(output_folder, exist_ok=True)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x_resized = cv2.resize(image, (W, H))  # Store resized image separately
        x_normalized = x_resized / 255.0
        x_expanded = np.expand_dims(x_normalized, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, (W, H))  # Store resized mask separately
        y_normalized = mask_resized / 255.0
        y_binary = y_normalized > 0.5
        y_int = y_binary.astype(np.int32)

        """ Prediction """
        y_pred = model.predict(x_expanded)[0]
        y_pred_binary = np.squeeze(y_pred, axis=-1) > 0.5
        y_pred_int = y_pred_binary.astype(np.int32)

        """ Saving the prediction images in the individual folder """
        cv2.imwrite(os.path.join(output_folder, f"{name}_original_image.png"), x_resized)
        cv2.imwrite(os.path.join(output_folder, f"{name}_mask.png"), mask_resized)
        cv2.imwrite(os.path.join(output_folder, f"{name}_predicted_mask.png"), y_pred_binary*255)

        """ Flatten the array """
        y = y_int.flatten()
        y_pred = y_pred_int.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")

