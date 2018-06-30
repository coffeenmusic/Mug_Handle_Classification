import os
import numpy as np
import tensorflow as tf
import random
import csv
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from microwave import Microwave
import cv2
import time

data_dir = 'Images/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))
	
lb = LabelBinarizer()
lb.fit(labels)

with tf.Session() as sess:	
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build(input_)
 
loader = tf.train.import_meta_graph('checkpoints/mugs.ckpt.meta')	
with tf.Session() as sess:	
    loader.restore(sess, tf.train.latest_checkpoint('checkpoints'))
	
    graph = tf.get_default_graph()
	
    inputs_ = graph.get_tensor_by_name("inputs:0")
    predicted = graph.get_tensor_by_name("predicted:0")
    
    count = 0
    fig = plt.figure(figsize=(15,20))
    plt.ion() # Interactive Mode
    plt.show()

    m = Microwave()
    while(count < 300):
        start_time = time.time()
        m.cam_capture()
        print(time.time() - start_time)
	
        img = utils.load_image_capture(m.rotated_img)
        img = img.reshape((1, 224, 224, 3))

        feed_dict = {input_: img}
        code = sess.run(vgg.relu6, feed_dict=feed_dict)	
	
        feed = {inputs_: code}
        prediction = sess.run(predicted, feed_dict=feed).squeeze()
	
	    # Plot image and class predictions-------------

        # Camera Capture
        plt.subplot(211)
        plt.imshow(m.rotated_img) # Show camera capture

        # Prediction Probabilities Bar Graph
        plt.subplot(212)
        plt.barh(np.arange(len(lb.classes_)), prediction)
        _ = plt.yticks(np.arange(len(lb.classes_)), lb.classes_)

        # Bool Prediction Represented In Figure Color
        if prediction[1] > 0.7:
            fig.patch.set_facecolor('xkcd:mint green');
        else:
            fig.patch.set_facecolor('red');

        plt.draw()
        plt.pause(0.01)
        plt.gcf().clear()
		
        count += 1


