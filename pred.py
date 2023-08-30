import os
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from glob import glob

H = 512
W = 512

""" Directory for storing prediction results """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model1.h5")

    """ Load the images for prediction """
    test_x = sorted(glob(os.path.join("test", "orignal_img", "*.png")))    #add your path of the image that you want to predict for

    print(f"Test: {len(test_x)}")

    """ Prediction and saving """
    for x in test_x:
        """ Extract the name """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Create a folder for the current prediction """
        output_folder = os.path.join("results", f"{name}_prediction")
        create_dir(output_folder)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x_resized = cv2.resize(image, (W, H))
        x_normalized = x_resized / 255.0
        x_expanded = np.expand_dims(x_normalized, axis=0)

        """ Prediction """
        y_pred = model.predict(x_expanded)[0]
        y_pred_binary = np.squeeze(y_pred, axis=-1) > 0.5

        """ Saving the predicted mask """
        cv2.imwrite(os.path.join(output_folder, f"{name}_predicted_mask.png"), y_pred_binary * 255)

        print(f"Processed: {name}")
