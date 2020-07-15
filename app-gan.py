# Importing Required Libraries
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from keras.models import load_model

from flask import Flask, send_file

# Define Paths and Constants

model_path_gen = 'model/model_gen.h5'

noise_dim = 100
image_dim = (32, 32)

# Debug var
debug = 1

def plot_gen(n_ex=1,dim=(4,4), figsize=(7,7) ):
    noise = np.random.normal(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)
    if debug == 1: print("Entered plot_gen Function")
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        if debug == 1: print("Entered FOR in plot_gen Function")
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]
        
    return img

generator = load_model(model_path_gen)

app = Flask(__name__)

@app.route('/')
def frontpage():
    return """
<!doctype html>
<head><title>dynamic</title></head>
<body>

<div>
<img style="border: 1px dotted red" src="/example1.png" />
</div>

</body>
</html>
"""

@app.route('/example1.png')
def example1():
    test = plot_gen(n_ex=1,dim=(1,1), figsize=(3,3))
    test = test*255
    img = Image.fromarray(test.astype('uint8'))
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    return nocache(send_file(file_object, mimetype='image/PNG'))

def nocache(response):
    """Add Cache-Control headers to disable caching a response"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate,
 max-age=0'
    return response
