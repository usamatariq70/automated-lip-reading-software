import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def pointsExtraction(fileName):

    cap = cv2.VideoCapture(fileName)
    dataOfFrames = list()
    frame_count = 0

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if frame_count == 29:
            break
        if ret is True:
            frame = cv2.resize(frame, (256,256))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                marks = np.zeros((2, 20), int)
                i = 0

                for n in range(48, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    marks[0, i] = x
                    marks[1, i] = y
                    i += 1
                dataOfFrames.append(np.array(marks))
                frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(frame_count)
    fileName = fileName.split('/')
    fileName = fileName[-1].split('.')
    np.save(fileName[0] + '.npy', dataOfFrames)
    return fileName[0]
