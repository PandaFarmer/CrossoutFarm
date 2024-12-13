from color_count import file_pixel_count_over_threshold

#Todo make version that works on different screens/settings?
def color_countTest():
    ColorInfos = {"BlueAlliedTriangle.png":{"Color": [ 69, 163, 255], "Threshold": 42},
        "EscortMissionIndicator.png":{"Color": , "Threshold": 42},
        "PurpleCrosshair.png":{"Color": , "Threshold": 42},
        "RedChevron.png":{"Color": , "Threshold": 42}}
    ColorMargin = 20
    CountMargin = 20
    for PictureFileName, ColorInfo in ColorInfos.items():
        with open(PictureFileName) as f:
            assert(file_pixel_count_over_threshold(f, ColorInfo["ColorHex"], ColorInfo["Threshold"]))
            assert(file_pixel_count_over_threshold(f, ColorInfo["ColorHex"] + ColorMargin, ColorInfo["Threshold"]))
            assert(not file_pixel_count_over_threshold(f, ColorInfo["ColorHex"] - ColorMargin, ColorInfo["Threshold"]))
            assert(file_pixel_count_over_threshold(f, ColorInfo["ColorHex"], ColorInfo["Threshold"] + CountMargin))
            assert(not file_pixel_count_over_threshold(f, ColorInfo["ColorHex"], ColorInfo["Threshold"] - CountMargin))
#BlueAlliedTriangle.png, dominant_color: [ 69 163 255], _count: 44
#EscortMissionIndicator.png, dominant_color: [255 150 0], _count: 630
#PurpleCrosshair.png, dominant_color: [191 114 252], _count: 48
#RedChevron.png, dominant_color: [255  50  50], _count: 80

def ObjectCrossCorrelationTest():
    pass

def OCRHuggingFaceTest():
    pass

def OCRPyTesseract():
    pass

def PausingTest():
    pass

def ScreenCropTest():
    pass

if __name__ == "__main__":
    pass