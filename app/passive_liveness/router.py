import asyncio
import base64

import cv2
import numpy as np
from fastapi import APIRouter
from loguru import logger

import app.passive_liveness.schemas as schemas

from app.passive_liveness.model import AntiSpoofPredict
from app.passive_liveness.crop import CropImage

from settings import settings


router = APIRouter(
    prefix='/passive_liveness',
    tags=['passive_liveness'],
)

path_to_caffemodel = settings.storage_folder.joinpath('detection_model/Widerface-RetinaFace.caffemodel')
path_to_deploy = settings.storage_folder.joinpath('detection_model/deploy.prototxt')

path_to_MiniFASNetV2 = settings.storage_folder.joinpath('anti_spoofing/2.7_80x80_MiniFASNetV2.pth')
path_to_MiniFASNetV1SE = settings.storage_folder.joinpath('anti_spoofing/4_0_0_80x80_MiniFASNetV1SE.pth')

model = AntiSpoofPredict(0, (path_to_deploy, path_to_caffemodel), (path_to_MiniFASNetV2, path_to_MiniFASNetV1SE))

MiniFASNetV2 = model.MiniFASNetV2
MiniFASNetV1SE = model.MiniFASNetV1SE

MiniFASNetV2_params = model.MiniFASNetV2_params
MiniFASNetV1SE_params = model.MiniFASNetV1SE_params

image_cropper = CropImage()

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

    image_bbox = model.get_bbox(camera_image)
    prediction = np.zeros((1, 3))

    MiniFASNetV2_params["org_img"] = camera_image
    MiniFASNetV2_params["bbox"] = image_bbox

    MiniFASNetV1SE_params["org_img"] = camera_image
    MiniFASNetV1SE_params["bbox"] = image_bbox

    face_for_MiniFASNetV2 = image_cropper.crop(**MiniFASNetV2_params)
    face_for_MiniFASNetV1SE = image_cropper.crop(**MiniFASNetV1SE_params)

    prediction_of_MiniFASNetV2 = model.predict(MiniFASNetV2, face_for_MiniFASNetV2)
    prediction_of_MiniFASNetV1SE = model.predict(MiniFASNetV1SE, face_for_MiniFASNetV1SE)

    prediction = prediction_of_MiniFASNetV2 + prediction_of_MiniFASNetV1SE
    label = np.argmax(prediction)

    if label == 1:
        status = schemas.FaceLivenessStatus.true
    else:
        status = schemas.FaceLivenessStatus.false

    logger.info(f'status is {status}')
    return schemas.FaceLivenessOutput(status=status)


async def main():
    camera_image_path = 'images.jpeg'

    camera_image: np.ndarray = cv2.imread(camera_image_path)
    image_bytes = cv2.imencode('.jpg', camera_image)[1].tobytes()
    camera_image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    input_data = schemas.FaceLivenessInput(
        camera_image_b64=camera_image_b64,
    )

    res = await passive_liveness(input_data)


if __name__ == '__main__':
    asyncio.run(main())
