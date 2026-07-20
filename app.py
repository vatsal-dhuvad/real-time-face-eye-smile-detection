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
            --app-bg: #f3f6fb;
            --panel-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #4b5563;
            --border: #d8e0ec;
            --accent: #2563eb;
            --green: #15803d;
            --red: #dc2626;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }
        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, label, span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }
        .main-title {
            color: var(--text-main);
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 1.4rem;
        }
        .info-box {
            background: #e8f5ee;
            border: 1px solid #b8e2c8;
            color: #14532d;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 1rem 0 1.2rem;
            font-weight: 700;
        }
        .legend {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1rem;
        }
        .legend span {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.45rem 0.65rem;
            font-weight: 700;
        }
        .face-label { color: #15803d !important; }
        .eye-label { color: #1d4ed8 !important; }
        .smile-label { color: #dc2626 !important; }
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
        div[data-testid="stCameraInput"] section,
        div[data-testid="stCameraInput"] div {
            background: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stCameraInput"] *,
        div[data-testid="stSlider"] *,
        div[data-testid="stNumberInput"] * {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploaderDropzone"] button,
        div[data-testid="stCameraInput"] button,
        div[data-testid="stDownloadButton"] button {
            background: #eff6ff !important;
            border: 1px solid #bfdbfe !important;
            color: #1d4ed8 !important;
            -webkit-text-fill-color: #1d4ed8 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        button[data-baseweb="tab"] p {
            color: var(--text-muted) !important;
            font-weight: 700;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--accent) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cascades():
    cascade_dir = Path(cv2.data.haarcascades)
    face = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
    smile = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_smile.xml"))

    if face.empty() or eye.empty() or smile.empty():
        raise RuntimeError("OpenCV Haar cascade files could not be loaded.")

    return face, eye, smile


def detect_face_eye_smile(image_bgr: np.ndarray, settings: dict[str, float | int]) -> tuple[np.ndarray, dict[str, int]]:
    face_cascade, eye_cascade, smile_cascade = load_cascades()
    output = image_bgr.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=float(settings["face_scale"]),
        minNeighbors=int(settings["face_neighbors"]),
        minSize=(int(settings["min_face_size"]), int(settings["min_face_size"])),
    )
    eye_count = 0
    smile_count = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (34, 197, 94), 3)
        cv2.putText(
            output,
            "Face",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (34, 197, 94),
            2,
        )

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = output[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=float(settings["eye_scale"]),
            minNeighbors=int(settings["eye_neighbors"]),
        )
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

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=float(settings["smile_scale"]),
            minNeighbors=int(settings["smile_neighbors"]),
        )
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

    return output, {"faces": len(faces), "eyes": eye_count, "smiles": smile_count}


class FaceEyeSmileProcessor(VideoProcessorBase):
    def __init__(self, settings: dict[str, float | int]):
        self.settings = settings

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        detected_image, _ = detect_face_eye_smile(image_bgr, self.settings)
        return av.VideoFrame.from_ndarray(detected_image, format="bgr24")


def image_file_to_bgr(image_file) -> np.ndarray:
    image = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
    image_rgb = np.array(image)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Could not create PNG result.")
    return buffer.tobytes()


st.markdown('<div class="main-title">Face, Eye and Smile Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detect faces, eyes, and smiles from a live camera, camera photo, or uploaded image.</div>',
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

    face_scale = st.slider("Face scale factor", 1.05, 1.50, 1.20, 0.05)
    face_neighbors = st.slider("Face accuracy", 3, 10, 5)
    min_face_size = st.slider("Minimum face size", 30, 180, 60, 10)

    eye_scale = st.slider("Eye scale factor", 1.05, 1.40, 1.10, 0.05)
    eye_neighbors = st.slider("Eye accuracy", 10, 35, 22)

    smile_scale = st.slider("Smile scale factor", 1.20, 2.20, 1.70, 0.05)
    smile_neighbors = st.slider("Smile accuracy", 10, 35, 22)

    st.header("Camera")
    st.caption("Live camera needs localhost or HTTPS browser permission.")

settings = {
    "face_scale": face_scale,
    "face_neighbors": face_neighbors,
    "min_face_size": min_face_size,
    "eye_scale": eye_scale,
    "eye_neighbors": eye_neighbors,
    "smile_scale": smile_scale,
    "smile_neighbors": smile_neighbors,
}

st.markdown(
    """
    <div class="legend">
        <span class="face-label">Face box</span>
        <span class="eye-label">Eye box</span>
        <span class="smile-label">Smile box</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Live Camera", "Camera Photo / Upload"])

with tabs[0]:
    st.subheader("Live Camera Detection")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="face-eye-smile-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: FaceEyeSmileProcessor(settings),
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
        output_bgr, counts = detect_face_eye_smile(input_bgr, settings)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Faces", counts["faces"])
        col_b.metric("Eyes", counts["eyes"])
        col_c.metric("Smiles", counts["smiles"])

        st.image(output_rgb, caption="Detection Result", use_container_width=True)
        st.download_button(
            "Download Result",
            data=bgr_to_png_bytes(output_bgr),
            file_name="face_eye_smile_detection_result.png",
            mime="image/png",
        )
