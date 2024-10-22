from PIL import ExifTags
import base64
from io import BytesIO
from PIL import Image

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





def convert_cropped_images_to_base64(cropped_images):
    """Function to convert cropped PIL images into base64-encoded strings"""
    base64_images = {}
    
    for view, img in cropped_images.items():
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
        base64_images[view] = img_str
    
    return base64_images
