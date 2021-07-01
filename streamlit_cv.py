import streamlit as st
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
from PIL import Image
import tempfile

DEMO_IMAGE = "demo.JPG"
DEMO_VIDEO = "demo.mp4"
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first_child{
        width: 350px
        }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first_child{
        width: 350px
        margin-left: -350px
        }
    </style>

    """, unsafe_allow_html=True
            )

st.sidebar.title("Меню")

st.sidebar.subheader("Выбор распознавания")

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


app_mode = st.sidebar.selectbox("Выберите тип распознавания",["Начальная страница","обнаружение лица","face mesh"])

if app_mode == "Начальная страница":
    vid = cv2.VideoCapture(0)
    vid.release()
    st.title("Давайте начнём")

    st.markdown("## На этом сайте вы можете ознакомиться с работой opencv")
    st.markdown("### Примеры:")
    st.video("C:/Users/alexm/Pictures/test_demo.mp4")

if app_mode == "обнаружение лица":
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Использовать камеру')
    record = st.sidebar.checkbox("Включить запись")
    if record:
        st.checkbox("Запись", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # max faces
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.markdown(' ## Вывод')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Загрузить видео", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0

    kpi1, kpi2, kpi3 = st.beta_columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    detector = FaceDetector(detection_confidence)
    while vid.isOpened():
        i += 1
        ret, frame = vid.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_count = 0
        frame, bboxs = detector.findFaces(frame)
        # currTime = time.time()
        # fps = 1 / (currTime - prevTime)
        # prevTime = currTime
        if record:
            # st.checkbox("Recording", value=True)
            out.write(frame)
        # Dashboard
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()

if app_mode == "face mesh":
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Использовать камеру')
    record = st.sidebar.checkbox("Включить запись")
    if record:
        st.checkbox("Запись", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # max faces
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.markdown(' ## Вывод')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Загрузить видео", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        vid.release()
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfflie.name)
    fps = 0
    i = 0


    detector = FaceMeshDetector(detection_confidence)
    while vid.isOpened():
        i += 1
        ret, frame = vid.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_count = 0
        frame, bboxs = detector.findFaceMesh(frame)
        # currTime = time.time()
        # fps = 1 / (currTime - prevTime)
        # prevTime = currTime
        if record:
            # st.checkbox("Recording", value=True)
            out.write(frame)
        # Dashboard
        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        frame = image_resize(image=frame, width=640)
        stframe.image(frame, channels='BGR', use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    vid.set(OPENCV_VIDEOIO_DEBUG=1)
    out.release()


# st.markdown("***hi***")