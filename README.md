# DeepForestVision

## Foreword
DeepForestVision is developed under CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0) by an academic team from the French Muséum National d'Histoire Naturelle (MNHN) as part of the One Forest Vision initiative (https://www.oneforestvision.org).

DeepForestVision is available in the AddaxAI interface (https://addaxdatascience.com/addaxai/) that can be run on Windows, Linux and MacOS without programming knowledge. This Github page provides the model weights and inference code.


## Using DeepForestVision

1) Install the dependencies from requirements.text. If the *PytorchWildlife* library fails to install *boto3*,
please install the dependencies *without* a virtual environment.
2) Run DFV.py to predict taxa from photos and videos using the following optional arguments:
   
**--data_dir** (str, default = './data' ): Folder where your photos and videos to process are stored (can be a mix of both, accepts subfolders)

**--predictions_dir** (str, default = './predictions'): Folder where you want the csv file with predictions to be stored (created automatically if non-existing)

**--detection_threshold** (float, default = .2): Detection score threshold above which MegaDetector detections are kept (created automatically)

**--stride** (float,  default = 1): Number of seconds between two extracted frames for videos

3) Predictions are stored in csv format in the *predictions* folder. They contain, for each file (photo or video): file path, file name, scores of class, prediction, confidence score.

## Examples

Standard use:

```python DFV.py```

With arguments:

```python DFV.py --detection_threshold .5 --stride .5 --data_dir '/home/documents/camera_trap_data' --predictions_dir '/home/documents/results'```
