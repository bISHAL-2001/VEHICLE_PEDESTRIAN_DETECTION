import time

import cv2
import numpy as np


class Detector:
    def __init__(self, video_path, config_path, model_path, classes_path):
        self.video_path = video_path  # sets the path for the video or the camera access
        self.config_path = config_path  # sets the model configuration path
        self.model_path = model_path  # sets the trained model path
        self.classes_path = classes_path  # sets the class path of the trained model

        self.classesList = []  # to store the classes
        self.colorList = []  # to store the colors

        # configuring the deep learning network
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(1000, 900)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.classes()

    def classes(self):

        # stores the classes in a list, from the classes file, as separate entities
        with open(self.classes_path, 'r') as f:
            self.classesList = f.read().splitlines()

        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        # print(self.classesList)

    def onVideo(self, capture_duration):
        cap = cv2.VideoCapture(self.video_path)  # To the read the video or the live footage

        if not cap.isOpened():
            print("Error opening file ...")
            return

        (success, image) = cap.read()

        startTime = 0
        time_elapsed = 0
        while success and time_elapsed <= capture_duration:
            print(f"{time_elapsed}, {capture_duration}")
            time_elapsed = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / int(cap.get(cv2.CAP_PROP_FPS)))
            # for the fps detection
            #
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime
            #

            # For extraction of the class IDs, Confidence and the dimension of objects from a particular frame
            classIds, confidences, bboxs = self.net.detect(image, confThreshold=0.6, nmsThreshold=1.5)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIds = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.1)

            if len(bboxIds) != 0:
                for i in range(len(bboxIds)):
                    bbox = bboxs[np.squeeze(bboxIds[i])]  # stores the list of x, y, width, height of a detected object
                    classConfidence = confidences[np.squeeze(bboxIds[i])]  # for the confidence of a detected object
                    classLabelID = np.squeeze(classIds[np.squeeze(bboxIds[i])])  # for the id of the detected object
                    classLabel = self.classesList[classLabelID]  # for name of the detected object
                    # classColor = [int(c) for c in self.colorList[classLabelID]] # For different color for different objects

                    # Text for object detected
                    displayText = "{}:{:.1f}%".format(classLabel, classConfidence * 100)

                    x, y, w, h = bbox  # to store the x, y coordinate and the width and height of the area of the object detected

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0),
                                  thickness=2)  # Bounding rectangle for identified object
                    cv2.rectangle(image, (x, y), (x + w, y - 22), color=(0, 0, 0),
                                  thickness=-1)  # Background for the detected object name
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

                    ####################################################################################################
                    ######################### Designing the Boundaries of the Bounding Frame ###########################
                    lineWidth = min(int(w * 0.2), int(h * 0.2))

                    # Upper left corner
                    cv2.line(image, (x, y), (x + lineWidth, y), (0, 0, 255), thickness=2)
                    cv2.line(image, (x, y), (x, y + lineWidth), (0, 0, 255), thickness=2)

                    # Upper right corner
                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), (0, 0, 255), thickness=2)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), (0, 0, 255), thickness=2)

                    # Lower left corner
                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), (0, 0, 255), thickness=2)
                    cv2.line(image, (x, y + h), (x, y + h - lineWidth), (0, 0, 255), thickness=2)

                    # Lower right corner
                    cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), (0, 0, 255), thickness=2)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), (0, 0, 255), thickness=2)

                    # total_objects_identified += 1

            # Background of the FPS text
            cv2.rectangle(image, (15, 20), (140, 58), color=(0, 0, 0), thickness=-1)
            # FPS text
            cv2.putText(image, "FPS: " + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", image)

            # Frame exit entry
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == 27 or cv2.getWindowProperty('Result', cv2.WND_PROP_VISIBLE) < 1:
                break
            (success, image) = cap.read()  # for successive frame captures

        cv2.destroyAllWindows()
