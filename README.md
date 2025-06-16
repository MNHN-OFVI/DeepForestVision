# DeepForestVision - reviewers only

## Foreword
DeepForestVision is developed under CC BY-NC-SA 4.0 license

DeepForestVision will be made publicly available in the AddaxAI interface (https://addaxdatascience.com/addaxai/),
that can be run on Windows, Linux and MacOS. Pending public release, we provide this Python code for reviewers to run the algorithm on Windows, Linux and MacOS.
It will also be made public on the project Github page.


## Using DeepForestVision

1) Make sure the dependencies in requirements.text are installed. If the *PytorchWildlife* library fails to install *boto3*,
please install the dependencies *without* a virtual environment.
2) Run DFV.py to predict taxa from photos and videos using the following optional arguments:
   
**--data_dir** (str, default = './data' ): Folder where your photos and videos to process are stored (can be a mix of both, accepts subfolders)

**--predictions_dir** (str, default = './predictions'): Folder where you want the csv file with predictions to be stored

**--detection_threshold** (float, default = .2): Detection score threshold above which MegaDetector detections are kept (created automatically)

**--stride** (float,  default = 1): Number of seconds between two extracted frames for videos
    
## Examples

Standard use:

```python DFV.py```

With arguments:

```python DFV.py --detection_threshold .5 --stride .5 --data_dir '/home/documents/camera_trap_data' --predictions_dir '/home/documents/results'```
