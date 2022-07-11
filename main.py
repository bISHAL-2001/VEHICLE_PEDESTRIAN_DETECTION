from Detect import *
import os


def main():
    # For object detection and object tracking in either video or live footage
    video_path = "Test_Video1.mp4"

    # For Model Configuration
    config_path = os.path.join("model_data", "config.pbtxt")

    # For the model path
    model_path = os.path.join("model_data", "model.pb")

    # For the classes trained in the trained model
    classes_path = os.path.join("model_data", "classes.names")

    detector = Detector(video_path, config_path, model_path, classes_path)
    detector.onVideo()


if __name__ == '__main__':
    main()
