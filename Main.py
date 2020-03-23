import sys

import time
from picamera.array import PiRGBArray
from picamera import PiCamera

import DetectChars
import DetectPlates

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)


def main(imgOriginalScene):

    if imgOriginalScene is None:
        print("error: image not read from file")
        sys.exit()

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

    listOfPossiblePlates = DetectChars.detectCharsInPlates(
        listOfPossiblePlates
    )

    if listOfPossiblePlates:
        listOfPossiblePlates.sort(
            key=lambda possiblePlate: len(possiblePlate.strChars),
            reverse=True
        )
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) in (7, 8):
            print(licPlate.strChars)

    return


if __name__ == "__main__":
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
    if blnKNNTrainingSuccessful is False:
        print("error: KNN traning was not successful")
        sys.exit()

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.vflip = True
    camera.hflip = True
    time.sleep(0.1)  # camera warmup

    while True:
        rawCapture = PiRGBArray(camera, size=(640, 480))
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        main(image)
        time.sleep(3)
