import cv2

version = '2.2.8'
counter = 0
video_cap = cv2.VideoCapture(f'flat_data/videos/{version}.mp4')
resolution = (640, 360)

while True:
    success, image = video_cap.read()

    if not success:
        break
    image = cv2.resize(image, resolution)
    cv2.imwrite(f'darkflow/dataset/frames/{version}-{counter}.jpg', image)
    counter += 1
