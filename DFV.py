import argparse
import os
import shutil
import cv2
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from PytorchWildlife.models import detection as pw_detection
from supervision import ImageSink
from supervision.utils import video as video_utils
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

from functions import clean_dir, list_photos_videos, save_cropped_images

# PARAMETERS
parser = argparse.ArgumentParser(description="Demo script with default arguments.")
parser.add_argument('--data_dir', type=str, default=Path('./data'), help='Folder with photos and videos to process')
parser.add_argument('--predictions_dir', type=str, default=Path('./predictions'), help='Folder where to save predictions')
parser.add_argument('--detection_threshold', type=float, default=.2, help='Detection score threshold above which MegaDetector detections are kept')
parser.add_argument('--stride', type=float, default=1, help='Stride of frames extraction for videos (in seconds)')
args = parser.parse_args()
data_dir, predictions_dir, detection_threshold, stride = args.data_dir, args.predictions_dir, args.detection_threshold, args.stride
images_max = 3000
extensions_photos = ('.jpg', '.JPG', '.jpeg', '.JPEG'', .png', '.PNG')
extensions_videos = ('.avi', '.AVI', '.mov', '.MOV', '.mp4', '.MP4')

# INITIALISATION OF DETECTOR (MegaDetector v5) AND CLASSIFIER (DINOv2 large)
device = "cuda" if torch.cuda.is_available() else "cpu"
detection_model = pw_detection.MegaDetectorV5(device=device, pretrained=True)
image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
classifier = AutoModelForImageClassification.from_pretrained('./classifier_weights')
pipe = pipeline('image-classification', model=classifier, image_processor= image_processor, device=device)
labels = classifier.config.id2label
taxons = list(labels.values())
taxons_count = len(taxons)
taxons_all = taxons + ['human', 'vehicle']


# DETECTION AND CLASSIFICATION
def predict_images(images_dir, detections_dir, data_dir, predictions):
    """
    Performs detection and classification on images and adds the results to the predictions dataframe.
    Args:
        images_dir (str):Path of the directory containing the images (either photos or frames extracted from videos).
        detections_dir (str): Path of the directory where to save the cropped detections.
        data_dir (str): Path of the data directory.
        predictions (dataframe): Dataframe containing previous predictions.
    Output:
        predictions (dataframe): Predictions dataframe updated with the new predictions.
    """
    print('Detecting...')
    detections_list = detection_model.batch_image_detection(data_path=images_dir, batch_size=32, det_conf_thres = detection_threshold)
    detections_dict = save_cropped_images(detections=detections_list, detections_dir =  detections_dir)
    detections_images = list_photos_videos(detections_dir, extensions_photos+extensions_videos)
    for i in tqdm(range(len(detections_images)), desc='Classifying...', unit='step', colour='yellow'):
        detection = detections_images[i]
        filename = detection.split('_', 3)[3][:-4]
        filepath = os.path.join(data_dir, filename)
        frame = detection.split('_')[2][1:]
        detection_info = detections_dict[detection]
        detection_class = detection_info[0]
        detection_score = detection_info[1]
        if detection_class == 1:
            detection_class_str = 'human'
            classification_scores = [0]*taxons_count + [1,0]
        elif detection_class == 2:
            detection_class_str = 'vehicle'
            classification_scores = [0]*taxons_count + [0,1]
        elif detection_class == 0:
            detection_class_str = 'animal'
            detection_image = Image.open(os.path.join(detections_dir, detection))
            # pred_all = pipe(image)
            inputs = image_processor(images = detection_image, return_tensors = 'pt').to(device)
            logits = pipe.model(**inputs).logits
            classification_scores = torch.nn.functional.softmax(logits, dim=-1)
            classification_scores = classification_scores.cpu().tolist()[0]+[0,0]
        prediction = pd.DataFrame([[filepath, filename, frame, detection_class_str, detection_score]+ list(detection_score*np.array(classification_scores))], columns=list(predictions.columns))
        predictions = pd.concat([predictions, prediction])
    clean_dir(images_dir)
    clean_dir(detections_dir)
    return predictions

images_dir = Path('./temp/images')
detections_dir = Path('./temp/detections')
clean_dir(images_dir)
clean_dir(detections_dir)
os.makedirs(predictions_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
predictions = pd.DataFrame(columns=['Filepath', 'Filename', 'Frame', 'Detection class', 'Detection score'] + taxons_all)
filepaths_all = []

# Loop though data subdirectories, detect and classify
print('USING DETECTION THRESHOLD '+ str(detection_threshold) + ' AND STRIDE '+ str(stride))
data_subdirs = [x[0] for x in os.walk(data_dir)]
for data_subir in data_subdirs:
    print('PROCESSING FOLDER ' + str(data_subir))
    images_count = 0
    files = list_photos_videos(data_subir, extensions_photos+extensions_videos)
    for i in tqdm(range(len(files)), desc='Extracting frames...', unit='step', colour='green'):
        file = files[i]
        filepath = os.path.join(data_subir, file)
        filepaths_all.append(filepath)
        if images_count > images_max:
            predictions = predict_images(images_dir = images_dir, detections_dir = detections_dir, data_dir = data_subir, predictions= predictions)
            images_count = 0
        if file.endswith(extensions_photos):
            shutil.copy(filepath, os.path.join(images_dir, 'F0_' + file+'.JPG'))
            images_count += 1
        else:
            video = cv2.VideoCapture(filepath)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            for index, frame in enumerate(video_utils.get_video_frames_generator(source_path=filepath, stride=int(stride*fps))):
                ImageSink(target_dir_path=images_dir, overwrite=False).save_image(image=frame,image_name='F'+str(idx)+'_'+file+'.JPG')
                idx += 1
                images_count += 1
    predictions = predict_images(images_dir = images_dir, detections_dir = detections_dir, data_dir = data_subir, predictions= predictions)
    images_count = 0
    # predictions.to_csv(os.path.join(predictions_dir, 'predictions_raw' + timestamp + '.csv'), index = False)


print('CONSOLIDATING PREDICTIONS')
# Get a single prediction for each file (photo or video) by a majority vote over the detections
gpby_dict = {}
gpby_dict['Filepath'] = 'first'
gpby_dict['Filename'] = 'first'
for taxon in taxons+['human', 'vehicle']:
    gpby_dict[taxon] = ['mean']
predictions_grouped = predictions.groupby('Filepath').agg(gpby_dict)
predictions_grouped['Prediction'] = predictions_grouped[taxons_all].apply(lambda x: 'blank' if sum(x)==0 else taxons_all[np.argmax(x)], axis=1)
predictions_grouped['Confidence score'] = predictions_grouped[taxons_all].apply(lambda x: 'blank' if sum(x)==0 else np.max(x), axis=1)
predictions_grouped.columns = [predictions_grouped.columns[i][0] for i in range(len(predictions_grouped.columns))]
# Predict as 'blank' photos/videos for which there were no detections
for filepath in filepaths_all:
    if filepath not in predictions_grouped['Filepath']:
        prediction_blank = pd.DataFrame([[filepath, filepath.rsplit(os.sep,1)[1]]+ [0]*(taxons_count+2) + ['blank', 1-detection_threshold]], columns=list(predictions_grouped.columns))
        predictions_grouped = pd.concat([predictions_grouped, prediction_blank])
predictions_grouped = predictions_grouped.reset_index(drop=True)
predictions_grouped = predictions_grouped.sort_values(by='Filepath', axis=0)
predictions_grouped.to_csv(os.path.join(predictions_dir, 'predictions_' + 'stride_' + str(stride) + '_thresh_' + str(detection_threshold)+ '_' + timestamp + '.csv'), index = False)
print('PREDICTIONS SUCCESSFULLY SAVED TO FOLDER: '+str(predictions_dir))