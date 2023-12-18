# -*- coding: utf-8 -*-
"""
You need to install the following before using this file
!pip install keras-image-helper
!pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
"""

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

interpreter = tflite.Interpreter(model_path='SCP-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(533, 400))

url = 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:eco,dpr_2.0/rockcms/2023-08/melanoma-pictures-mc-230804-02-398f58.jpg'
X = preprocessor.from_url(url)

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)

classes = [
    'Melanoma',
    'Nevus',
    'Seborrheic_Keratosis',
    ]

dicresult = dict(zip(classes, preds[0]))
print(dicresult)