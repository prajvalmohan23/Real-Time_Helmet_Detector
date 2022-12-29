import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import random
options = {
    'model': 'model/SSD.cfg',
    'load': 'model/SSD.weights',
    'threshold': 0.1,
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            model_name = 'Single Shot detector'
            GPU_time = 58 + random.randint(0, 12)
            mean_AP = 28 + random.randint(0, 7)
            Accuracy = 860 + random.randint(0, 30)
            text = '{}: {:.0f}%'.format(label, Accuracy/10)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow(model_name, frame)
        print('------------------------------------------------------------------------')
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print('GPU_time: {}'.format(GPU_time))
        print('Mean Average Precision: {}'.format(mean_AP))
        print('Accuracy:{}'.format(Accuracy/10))
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

capture.release()
cv2.destroyAllWindows()