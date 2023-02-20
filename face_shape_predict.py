import cv2
import dlib

shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

frame = cv2.imread("/home/ubuntu/tmp/sd-agp-eval/2102.jpg")
shape = shape_predictor(frame, dlib.rectangle(0, 0, frame.shape[1], frame.shape[0]))

for j in range(68):
    x, y = shape.part(j).x, shape.part(j).y
    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

cv2.imshow('window0', frame)
cv2.waitKey(10000)
