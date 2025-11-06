import cv2
import os
from PIL import Image
import numpy as np
import shutil
from supervision import ImageSink, crop_image


def list_photos_videos(dir_path, extensions):
    """
    Lists all photos and videos within a directory.
    Args:
        dir_path (str): Path of the directory containing the photos and videos.
        extensions (list): List of allowed photo and video extensions.
    Output:
        photos_videos (list): List of all filenames of all photos and videos within the directory.
    """
    photos_videos = []
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith('.') and f.endswith(extensions):
            photos_videos += [f]
    return photos_videos

def clean_dir(dir_path):
    """
    Erases directory if it exists and creates a new empty one.
    Arg:
        dir_path (str): Path of the directory to clean.
    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def save_cropped_images(detections, detections_dir):
    """
    Saves cropped images based on the detection bounding boxes.
    Args:
        detections (list): Detection results containing image ID and detections.
        detections_dir (str): Directory where to save the cropped images.
    Outputs:
        detections_dict (dict): Dictionary mapping each detection filename to the detection's class (animal, human or vehicle) and its detection score.
        frames_dict (dict): Dictionary counting the number of detections by frame.

    """
    detections_dict = {}
    frames_dict = {}
    with ImageSink(target_dir_path=detections_dir, overwrite=False) as sink:
        for entry in detections:
            for i, (xyxy, _, detection_score, detection_class, _, _) in enumerate(entry["detections"]):
                image_cropped = crop_image(
                    image=np.array(Image.open(entry["img_id"]).convert("RGB")), xyxy=xyxy
                )
                image_name = "{}_{}_{}".format(
                    detection_class, i, entry["img_id"].rsplit(os.sep, 1)[1])
                sink.save_image(
                    image=cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR),
                    image_name=image_name,
                    ),
                detections_dict[image_name] = [detection_class, detection_score]
                frames_dict[entry["img_id"].split('/')[-1]] = frames_dict.get(entry["img_id"].split('/')[-1], 0)+1
    return detections_dict, frames_dict