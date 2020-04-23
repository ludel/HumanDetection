import time

import cv2
from darkflow.net.build import TFNet


def draw_bounding_boxes(frame, predictions):
    for prediction in predictions:
        frame = cv2.rectangle(
            img=frame,
            pt1=(prediction["topleft"]["x"], prediction["topleft"]["y"]),
            pt2=(prediction["bottomright"]["x"], prediction["bottomright"]["y"]),
            color=[0, 0, 255],
            thickness=1,
        )
    return frame


def main(
        video_file_path: str,
        fps: float,
        width: int,
        height: int,
        sensibility: float,
        analyzed_frames_frequency: int,
):
    model = TFNet(
        {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", 'threshold': sensibility}
    )
    model.load_from_ckpt()

    #model = TFNet(
    #    {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", 'threshold': sensibility}
    #)
    print(f"Reading input video ...")
    start = time.time()
    capture = cv2.VideoCapture(video_file_path)
    print(fps, width, height)
    video = cv2.VideoWriter(
        "./video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print("Detecting poachers ...")
    success, count, predictions = True, 0, []
    while success:
        success, frame = capture.read()
        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            if count % int(fps / analyzed_frames_frequency) == 0:
                predictions = [
                    prediction
                    for prediction in model.return_predict(frame)
                    if prediction["label"] == "person"
                ]
            for prediction in predictions:
                frame = cv2.rectangle(
                    img=frame,
                    pt1=(prediction["topleft"]["x"], prediction["topleft"]["y"]),
                    pt2=(
                        prediction["bottomright"]["x"],
                        prediction["bottomright"]["y"],
                    ),
                    color=[0, 0, 255],
                    thickness=1,
                )
            video.write(frame)
        count += 1
    print(f"{count} frames")
    print("Releasing video ...")

    video.release()
    print(f"{round(time.time() - start, 2)} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path of the input video where to detect poachers"
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="The number of frame per second of the input video",
        type=float,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="The wanted width of the output video",
        choices=[640],
        type=int,
    )
    parser.add_argument(
        "-e",
        "--height",
        help="The wanted height of the output video",
        choices=[360],
        type=int,
    )
    parser.add_argument(
        "-s",
        "--sensibility",
        help="The sensibility of of the detection, 0 is the most sensible,"
             " +inf is the least sensible (will detect nothing)",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-a",
        "--analyzed",
        help="The number of frames analyzed per second",
        type=float,
        default=5,
    )
    args = parser.parse_args()
    main(args.input, args.fps, args.width, args.height, args.sensibility, args.analyzed)
