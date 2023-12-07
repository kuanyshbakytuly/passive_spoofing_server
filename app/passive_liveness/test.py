import os
import cv2
import numpy as np

from app.passive_liveness.model import AntiSpoofPredict
from utils import parse_model_name
from app.passive_liveness.crop import CropImage



def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))

    # sum the prediction from single model's result
    for i, model_name in enumerate(os.listdir(model_dir)):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        face = image_cropper.crop(**param)
        this_predict = model_test.predict(face, os.path.join(model_dir, model_name))

        print(f"Infer {i}: {this_predict}")
        prediction += this_predict

    print("=============-.-=============")

    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)


    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)

    cv2.imwrite( result_image_name, image)

test(image_name= '/Users/kuanyshbakytuly/Desktop/Relive/silent_face_api/images/photo_2023-12-06 18.55.23.jpeg', model_dir='/Users/kuanyshbakytuly/Desktop/Relive/silent_face_api/model_pth/anti_spoofing', device_id=0)