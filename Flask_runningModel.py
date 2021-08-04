from flask import Flask , render_template, request
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import cv2
import os
import base64



app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict' , methods=['POST','GET'])
def home():   
    model = tf.keras.models.load_model('./o.model',
                                   custom_objects=None,
                                   compile=True)
    fileitem = request.files['filename']
    fileitem.save(fileitem.filename)
# check if the file has been uploaded
    #if fileitem.filename:
    # strip the leading path from the file name
     #   fn = os.path.basename(fileitem.filename)
      
   # open read and write the file into the server
    #open(fn, 'wb').write(fileitem.files.read())

    img1_color=[]
    img1=img_to_array(load_img(fileitem.filename))
    img1 = resize(img1 ,(256,256))
    img1_color.append(img1)
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    output1 = model.predict(img1_color)
    output1 = output1*128
    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]
 
    #cv2.imshow('result',lab2rgb(result))
    imsave("result.png", lab2rgb(result))
    imsave("./templates/result.png", lab2rgb(result))
    #cv2.waitKey(0)
    #data_uri = base64.b64encode(open('result.png', 'rb').read()).decode('utf-8')
    #img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    #print(img_tag)
    return render_template('result.html', data= './result.png')



if __name__ == "__main__":
    app.run(debug=True)