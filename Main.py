import cv2
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

    if len(listOfPossiblePlates) == 0:
        print("no license plates were detected")
    else:
        listOfPossiblePlates.sort(
            key=lambda possiblePlate: len(possiblePlate.strChars),
            reverse=True
        )
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("no characters were detected")
        else:
            print("license plate read from image = " + licPlate.strChars)

    print("----------------------------------------")

    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(
        p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(
        p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(
        p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(
        p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(
        licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(
            round(intPlateCenterY)) + int(round(plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(
            round(intPlateCenterY)) - int(round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

    cv2.putText(
        imgOriginalScene,
        licPlate.strChars,
        (
            ptLowerLeftTextOriginX,
            ptLowerLeftTextOriginY
        ),
        intFontFace,
        fltFontScale,
        SCALAR_YELLOW,
        intFontThickness
    )


if __name__ == "__main__":
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
    if blnKNNTrainingSuccessful is False:
        print("error: KNN traning was not successful\n")
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
