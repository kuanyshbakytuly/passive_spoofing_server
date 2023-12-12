import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from app.passive_liveness.utils import MiniFASNetV1SE, MiniFASNetV2, get_kernel, parse_model_name
from torchvision import transforms
from app.passive_liveness.utils import parse_model_name, Compose, ToTensor

MODEL_MAPPING = {
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
}

class Detection:
    def __init__(self, deploy, caffe):  
        self.detector = cv2.dnn.readNetFromCaffe(prototxt=str(deploy), caffeModel=str(caffe))
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]

        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id, path_to_detector, path_to_passive):
        super(AntiSpoofPredict, self).__init__(*path_to_detector)

        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        
        self.MiniFASNetV2, self.MiniFASNetV2_params = self._load_model(model_path=str(path_to_passive[0]))
        self.MiniFASNetV1SE, self.MiniFASNetV1SE_params = self._load_model(model_path=str(path_to_passive[1]))

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        if scale is None:
            param["crop"] = False


        kernel_size = get_kernel(h_input, w_input,)

        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        return model, param


    def predict(self, model, img):
        test_transform = Compose([
            ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            result = model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result