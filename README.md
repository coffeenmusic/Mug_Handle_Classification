# Stop Microwave With Handle Facing the Door
![Microwave Classifier](https://raw.githubusercontent.com/coffeenmusic/Mug_Handle_Classification/master/microwave_handle_classification.gif)

### Sample.py
Launches a GUI that can be used to create classification images. Will need an Images directory.

### Network.ipynb
This jupyter notebook can be used to train on the image data set given good classication images. Will need a checkpoints directory.

### WebCam_Predict.py
This will launch a matplotlib figure that will display captures from an attached webcam that is mounted in the microwave behind the mesh.
As the camera updates captures, the figure subplot will show predictions of whether the handle is facing the door.
The background color will display green if the handle is facing the door, otherwise it will display red.