import datetime

import cv2
from darkflow.net.build import TFNet


class Model:
    def __init__(self, **kwargs):
        self.tf_net = TFNet(kwargs)
        self.resolution = (640, 360)
        self.fps = 29.9
        print('Start time ', datetime.datetime.now())

    def train(self):
        self.tf_net.train()
        print('Train end time ', datetime.datetime.now())

    def draw_box(self, input_video, output_video, limit=300):
        counter = 0
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_video, four_cc, self.fps, self.resolution)
        video_input = cv2.VideoCapture(input_video)

        while True:
            success, frame = video_input.read()
            frame = cv2.resize(frame, self.resolution)

            if not success or counter > limit:
                break

            results = self.tf_net.return_predict(frame)
            for res in results:
                if res['label'] == 'person':
                    frame = cv2.rectangle(
                        frame, (res['topleft']['x'], res['topleft']['y']),
                        (res['bottomright']['x'], res['bottomright']['y']),
                        [0, 0, 255], 1
                    )

            counter += 1
            video_out.write(frame)

        video_out.release()


if __name__ == '__main__':
    options = {
        'model': 'cfg/tiny-yolo-1c.cfg',
        'load': 'bin/tiny-yolo-v1.1.weights',
        'threshold': 0.3,
        'train': True,
        'batch': 20,
        'epoch': 10,
        'annotation': 'dataset/annotations',
        'dataset': 'dataset/frames',
        'save': 150000,
        'height': 360,
        'width': 640,
        'gpu': 0.9,
    }

    model = Model(**options)
    model.train()

    print('Generate video ... ')
    model.draw_box('../flat_data/videos/2.2.9.mp4', '../output/predict-tiny-2.2.9-enormous_train.mp4', limit=1000)
