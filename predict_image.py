# USAGE python predict_image.py --image image_from_local|--url url_download_image


import argparse
import pickle

import cv2
import numpy as np
import requests
from multi_label_model import get_cnn_model

INPUT_SHAPE = (299, 299, 3)
IMAGE_DIMS = (299, 299, 3)

MODEL_PATH = r'model6/model6.h5'
LABEL_BIN_PATH = r'mlb.pkl'
# Khởi tạo ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="input image from local")
ap.add_argument("-u", "--url", required=False,
                help="url to download image")
args = vars(ap.parse_args())

# Load model và multilabel
print("[INFO] loading network...")
model = get_cnn_model()
model.load_weights(MODEL_PATH)
# model = load_model(r'model.020.h5')

mlb = pickle.loads(open(LABEL_BIN_PATH, "rb").read())


# read image
def _download_image(url):
    resp = requests.get(url)
    img = np.asarray(bytearray(resp.content), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _get_image(url):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# dự báo image
def _predict_image(image, model, mlb):
    # Lấy kích thước 3 kênh của image
    (w, h, c) = image.shape
    # Nếu resize width = 400 thì height resize sẽ là
    height_rz = int(h * 400 / w)
    # Resize lại ảnh để hiện thị
    output = cv2.resize(image, (height_rz, 400))
    # Resize lại ảnh để dự báo
    image = cv2.resize(image, IMAGE_DIMS[:2])
    # Dự báo xác suất của ảnh
    prob = model.predict(np.expand_dims(image, axis=0))[0]
    print(prob)
    # Trích ra 2 xác suất cao nhất
    argmax = np.argsort(prob)[::-1][:2]
    print(argmax)
    # Show classes và probability ra ảnh hiển thị
    for (i, j) in enumerate(argmax):
        # popup nhãn và xác suất dự báo lên ảnh hiển thị
        label = "{}: {:.2f}%".format(mlb.classes_[j], prob[j] * 100)
        cv2.putText(output, label, (5, (i * 20) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 2)
    # show the output image
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imshow("Output", output)
    cv2.waitKey(0)


if args['image']:
    image = _get_image(args['image'])
    _predict_image(image, model, mlb)
elif args['url']:
    print('downloading image .....')
    image = _download_image(args['url'])
    _predict_image(image, model, mlb)
else:
    print('please enter image or url')
