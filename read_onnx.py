import onnxruntime
import cv2
import numpy as np
import time

model_file = '/home/nvidia/procedure/onnx/add_argmax_layers.onnx'

with open(model_file,'rb') as f:
    model = f.read()

sess = onnxruntime.InferenceSession(model)


start_time = time.time()
for i in range(10):
    img = cv2.imread('/home/nvidia/procedure/keras/JPEGImages/1.jpg')
    img = cv2.resize(img,(224,224), interpolation= cv2.INTER_AREA)
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.reshape((1,224,224,3))
    pre  = sess.run(None, {'input_1':img})
    print(i)

end_time = time.time()
print('time:', end_time - start_time)
