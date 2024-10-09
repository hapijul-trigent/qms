import numpy as np
import cv2
from PIL import Image, ExifTags
import supervision as sv
from src.checklist import update_CHECKLIST, TOP_CHECKLIST, SIDE_CHECKLIST, BOTTOM_CHECKLIST

def correct_image_orientation(image):
    """Correct the orientation of an image based on its EXIF data."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def top_view_checks(image, model):
    """Perform checks for the top view."""
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .6]

    if len(detections.xyxy) == 0:
        for thing in TOP_CHECKLIST:
            update_CHECKLIST(thing, False, TOP_CHECKLIST)
    else:
        for thing in TOP_CHECKLIST:
            update_CHECKLIST(thing, detections.data['class_name'][0], TOP_CHECKLIST)

    return result.plot()

def side_view_checks(image, view_name, model):
    """Perform checks for the side view."""
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .6]

    if len(detections.xyxy) == 0:
        for thing in SIDE_CHECKLIST:
            update_CHECKLIST(thing, False, SIDE_CHECKLIST)
    else:
        for thing in SIDE_CHECKLIST:
            if thing in detections.data['class_name']:
                update_CHECKLIST(thing, True, SIDE_CHECKLIST)
            else:
                update_CHECKLIST(thing, False, SIDE_CHECKLIST)

    return result.plot()

def bottom_view_checks(image, model):
    """Perform checks for the bottom view."""
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .6]

    if len(detections.xyxy) == 0:
        for thing in BOTTOM_CHECKLIST:
            update_CHECKLIST(thing, False, BOTTOM_CHECKLIST)
    else:
        for thing in BOTTOM_CHECKLIST:
            update_CHECKLIST(thing, detections.data['class_name'][0], BOTTOM_CHECKLIST)

    return result.plot()

def merge_side_view_analysis(images, annotation_view_panels):
    for view_name, image in images.items():
        if image:
            annotated_view = side_view_checks(image, view_name, model=model_side_QA)
            annotation_view_panels[view_name].image(annotated_view, channels='bgr')
    return True
