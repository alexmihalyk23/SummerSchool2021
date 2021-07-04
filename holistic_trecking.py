import cv2
import mediapipe as mp
import pyvirtualcam

#
#
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
    cap = cv2.VideoCapture(0)
    img = np.zeros((720, 1280, 3), np.uint8)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 180)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, landmark_drawing_spec=
                drawing_spec)
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # StackedImge = cvzone.Utils.stackImages([image, img], 2,1)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cam.send(cv2.resize(frame, (1280, 720)))
            cam.sleep_until_next_frame()
