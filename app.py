from functools import lru_cache
from io import BytesIO
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer


st.set_page_config(
    page_title="Face Eye Smile Detection App",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            color-scheme: light !important;
        }
        :root {
            --app-bg: #f4f7fb;
            --panel-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #4b5563;
            --border: #d7dee8;
            --accent: #2563eb;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background: var(--app-bg);
            color: var(--text-main);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, label, span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text-main);
        }
        .main-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .info-box {
            background: #e8f5ee;
            border: 1px solid #b8e2c8;
            color: #14532d;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 1rem 0;
            font-weight: 600;
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetric"] * {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] div,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stNumberInput"] * {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploaderDropzone"] button {
            background: #eff6ff !important;
            border: 1px solid #bfdbfe !important;
            color: #1d4ed8 !important;
            -webkit-text-fill-color: #1d4ed8 !important;
        }
        button[data-baseweb="tab"] p {
            color: var(--text-muted) !important;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--accent) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@lru_cache(maxsize=1)
def load_cascades():
    cascade_dir = Path(cv2.data.haarcascades)
    face = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
    smile = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_smile.xml"))

    if face.empty() or eye.empty() or smile.empty():
        raise RuntimeError("OpenCV Haar cascade files could not be loaded.")

    return face, eye, smile


def detect_face_eye_smile(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    face_cascade, eye_cascade, smile_cascade = load_cascades()
    output = image_bgr.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    eye_count = 0
    smile_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.putText(
            output,
            "Face",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (34, 197, 94),
            2,
        )

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = output[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=22)
        eye_count += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (37, 99, 235), 2)
            cv2.putText(
                roi_color,
                "Eye",
                (ex, max(ey - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (37, 99, 235),
                1,
            )

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        smile_count += len(smiles)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (220, 38, 38), 2)
            cv2.putText(
                roi_color,
                "Smile",
                (sx, max(sy - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 38, 38),
                1,
            )

    counts = {
        "faces": len(faces),
        "eyes": eye_count,
        "smiles": smile_count,
    }
    return output, counts


class FaceEyeSmileProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        detected_image, _ = detect_face_eye_smile(image_bgr)
        return av.VideoFrame.from_ndarray(detected_image, format="bgr24")


def image_file_to_bgr(uploaded_file) -> np.ndarray:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    image_rgb = np.array(image)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


st.markdown('<div class="main-title">Face, Eye and Smile Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Open your camera or upload an image to detect faces, eyes, and smiles using OpenCV Haar cascades.</div>',
    unsafe_allow_html=True,
)

try:
    load_cascades()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Detection")
    st.caption("Green = Face, Blue = Eye, Red = Smile")
    st.header("Camera")
    st.caption("Use HTTPS or localhost for browser camera permission.")

st.markdown(
    '<div class="info-box">This app works with live browser camera using WebRTC. You can also test with a captured or uploaded image.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["Live Camera", "Image Test"])

with tabs[0]:
    st.subheader("Live Camera Detection")
    st.caption("Click Start, allow camera permission, and the app will draw detection boxes on the video.")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="face-eye-smile-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=FaceEyeSmileProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tabs[1]:
    st.subheader("Image Detection")
    source = st.radio("Image source", ["Take Photo", "Upload Image"], horizontal=True)

    image_input = None
    if source == "Take Photo":
        image_input = st.camera_input("Take a photo")
    else:
        image_input = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if image_input is not None:
        input_bgr = image_file_to_bgr(image_input)
        output_bgr, counts = detect_face_eye_smile(input_bgr)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Faces", counts["faces"])
        col_b.metric("Eyes", counts["eyes"])
        col_c.metric("Smiles", counts["smiles"])

        st.image(output_rgb, caption="Detection Result", use_container_width=True)
