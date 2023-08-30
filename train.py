import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob #extract the path
from sklearn.utils import shuffle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from model import build_unet
from keras import metrics
from metrics import dice_loss, dice_coef, iou



H = 512
W = 512

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

# def shuffling(x, y):
#     x, y = shuffle(x, y, random_state=42)
#     return x, y

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "Train", "images", "*.png")))
    train_y = sorted(glob(os.path.join(path, "Train", "mask_img_labelsTrain", "*.png")))
    valid_x = sorted(glob(os.path.join(path, "Val", "images", "*.png")))
    valid_y = sorted(glob(os.path.join(path, "Val", "mask_img_labelsval", "*.png")))
    print(len(train_x),len(train_y))
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0  #normalising
    x = x > 0.5   #threshhold to convert 0 and 1
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(x, y, batch=8):   #x y list of images
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 32
    lr = 1e-4
    num_epochs = 1
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Dataset """
    dataset_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/braincompute/code/Users/raunak.12007198/Splitted_Data_final/Splitted_Data_final"
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path)



    # train_path = os.path.join(dataset_path, "train")
    # valid_path = os.path.join(dataset_path, "valid")

    # train_x, train_y = load_data(train_path)
    # train_x, train_y = shuffling(train_x, train_y)
    # valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y,batch=batch_size) #batch=batch_size
    valid_dataset = tf_dataset(valid_x, valid_y,batch=batch_size)

    """ Model """
    model = build_unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )