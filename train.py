import argparse
import os
import pickle

import cv2
import glob2
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from multi_label_model import get_cnn_model

# Thiết lập ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
# ap.add_argument("-m", "--model", required=True, default="model_fashion_multitask_learning.h5",
#                 help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Thiết lập tham số
INPUT_SHAPE = (299, 299, 3)
N_CLASSES = 6
DATASET_DIR = r'D:\DoAnAI\Dataset'
BATCH_SIZE = 64

# Khởi tạo dữ liệu
images = []
labels = []


# Dataset


def _list_images(root_dir, exts=['.jpg', '.jpeg', '.png', '.jfif']):
    list_images = glob2.glob(r'D:\DoAnAI\Dataset' + '/**')
    image_links = []
    for image_link in list_images:
        for ext in exts:
            if ext in image_link[-5:]:
                image_links.append(image_link)
    return image_links


def _data_source(root_dir):
    # Lấy dữ liệu ảnh lưu vào pandas dataframe: label là nhãn, source là link tới ảnh
    imagePaths = sorted(_list_images(root_dir=root_dir))
    labels = [path.split("\\")[3] for path in imagePaths]
    data = pd.DataFrame({'label': labels, 'source': imagePaths})
    return data


data = _data_source(DATASET_DIR)
print(data)
data_sources = data.groupby('label').source.apply(lambda x: list(x))
print(data_sources)
for i, sources in enumerate(data_sources):
    np.random.shuffle(list(sources))
    label = data_sources.index[i]
    sources = data_sources[label]
    for imagePath in sources:
        # Đọc dữ liệu ảnh
        try:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, INPUT_SHAPE[:2])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            images.append(image)
        except Exception as e:
            print(e)
            os.remove(imagePath)
            print('remove' + imagePath)
        # Gán dữ liệu label
        gender, sentiment = label.split("_")
        labels.append([gender, sentiment])

# MultiLabel Encoding cho nhãn của ảnh
mlb = MultiLabelBinarizer()
# One-hot encoding cho fashion
y = mlb.fit_transform(labels)
# Lưu the multi-label binarizer
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

print('[INFO] classes of labels: ', mlb.classes_)
# Phân chia train/validation theo tỷ lệ 70/30
(X_train, X_val, y_train, y_val) = train_test_split(images, y,
                                                    test_size=0.3, random_state=123)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)


print('[INFO] X_train.shape: {}, y_train.shape: {}'.format(len(X_train), len(y_train)))
print(f"y_train label shape: {len(y_train[0])}")
print('[INFO] X_val.shape: {}, y_val.shape: {}'.format(len(X_val), len(y_val)))

# Khởi tạo data augumentation
image_aug = ImageDataGenerator(rotation_range=25,
                               width_shift_range=0.1, height_shift_range=0.1,
                               shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, fill_mode="nearest")
# image_aug.fit(X_train)

print('[INFO] training model...')

model = get_cnn_model()
# Khởi tạo optimizer
EPOCHS = 100
LEARNING_RATE = 0.00001
opt = keras.optimizers.Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

save_dir = 'ModelCheckpoint3'
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                             monitor='val_acc',
                                             verbose=1,
                                             save_weights_only=False,
                                             save_best_only=False)

# Huấn luyện mô hình
history = model.fit(
    # image_aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS, verbose=1, callbacks=[checkpoint])
np.save('history3.npy', history.history)
# # Lưu mô hình
print("[INFO] serializing network...")
model.save("model12.h5")

# lưu multi-label binarizer
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()


