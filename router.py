import asyncio
import base64

import cv2
import numpy as np
import os
from fastapi import APIRouter, HTTPException
from loguru import logger

import schemas as schemas

from model import AntiSpoofPredict
from utils import parse_model_name
from crop import CropImage



router = APIRouter(
    prefix='/passive_liveness',
    tags=['passive_liveness'],
)


@router.post(
    "/verify",
    response_model = schemas.FaceLivenessOutput
)

async def passive_liveness(
        face_liveness_input: schemas.FaceLivenessInput,
):
    camera_image_b64: str = face_liveness_input.camera_image_b64

    camera_image: np.ndarray = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(base64.b64decode(camera_image_b64), np.uint8), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )

    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(camera_image)
    prediction = np.zeros((1, 3))

    model_dir = "/Users/kuanyshbakytuly/Desktop/Relive/silent_face_api/model_pth/anti_spoofing"
    
    # sum the prediction from single model's result
    for i, model_name in enumerate(os.listdir(model_dir)):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": camera_image,
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
        prediction += this_predict


    label = np.argmax(prediction)

    if label == 1:
        status = schemas.FaceLivenessStatus.true
    else:
        status = schemas.FaceLivenessStatus.false

    logger.info(f'status is {status}')
    return schemas.FaceLivenessOutput(status=status)


async def main():
    camera_image_path = 'images/photo_2023-12-06 18.55.23.jpeg'

    camera_image: np.ndarray = cv2.imread(camera_image_path)
    image_bytes = cv2.imencode('.jpg', camera_image)[1].tobytes()
    camera_image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    input_data = schemas.FaceLivenessInput(
        camera_image_b64=camera_image_b64,
    )

    res = await passive_liveness(input_data)


if __name__ == '__main__':
    asyncio.run(main())
