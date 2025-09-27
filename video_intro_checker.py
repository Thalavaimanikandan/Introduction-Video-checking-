import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import dlib
import imageio
import torch

try:
    import whisper
except ImportError:
    raise ImportError("Install openai-whisper: pip install -U openai-whisper")

MAX_SIZE_MB = 200
MIN_DURATION = 5
MAX_DURATION = 120
MAX_SAMPLED_FRAMES = 20         
DOWNSAMPLE_WIDTH = 240          
FACE_TOLERANCE = 0.6
TRANSCRIBE_SECONDS = 25        
WHISPER_MODEL = "base"          
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"

INTRO_KEYWORDS = [
    "my name", "i am", "introduce", "i belong",
    "i completed", "i studied", "i graduated",
    "currently i am", "i have done"
]
CAPTION_KEYWORDS = ["intro", "myself", "self-introduction"]
COMPANY_KEYWORDS = [
    "company", "organization", "firm", "business", "enterprise",
    "corporate", "mnc", "multinational", "industry", "office", "workplace"
]
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

def run_ffmpeg_extract_segment(input_path: str, out_path: str, seconds: int) -> None:
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-t", str(seconds), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def get_video_properties(file_path: str) -> Tuple[int, float, float]:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = frames / fps if fps else 0.0
    cap.release()
    return frames, fps, duration


def transcribe_first_n_seconds(file_path: str, n_seconds: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "segment.wav")
        run_ffmpeg_extract_segment(file_path, audio_path, n_seconds)
        result = whisper_model.transcribe(audio_path, fp16=(device == "cuda"))
        return (result.get("text", "") or "").strip().lower()


def sample_frame_indices(total_frames: int, max_samples: int) -> List[int]:
    if total_frames <= max_samples:
        return list(range(total_frames))
    step = total_frames / float(max_samples)
    return [int(i * step) for i in range(max_samples)]


def encode_faces(frame_bgr: np.ndarray, num_jitters: int = 1) -> list[np.ndarray]:
    if frame_bgr is None:
        return []
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (DOWNSAMPLE_WIDTH, int(img.shape[0] * DOWNSAMPLE_WIDTH / img.shape[1])))
    detections = detector(resized, 1)
    encodings = []
    for det in detections:
        try:
            shape = sp(resized, det)
            face_descriptor = facerec.compute_face_descriptor(resized, shape, num_jitters)
            encodings.append(np.array(face_descriptor))
        except:
            continue
    return encodings

def check_video_info(file_path: str) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, "File not found"
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    _, _, duration = get_video_properties(file_path)
    
    if size_mb > MAX_SIZE_MB:
        return False, f"Video too large: {size_mb:.2f} MB"
    
    if duration < MIN_DURATION:
        return False, f"Video too short: {duration:.2f}s"
    
    if duration > MAX_DURATION:
        if size_mb > MAX_SIZE_MB:
            return False, f"Video too long and large ({duration:.2f}s, {size_mb:.2f}MB)"
        else:
            return True, f"Video duration > {MAX_DURATION}s but size is ok ({duration:.2f}s, {size_mb:.2f}MB)"
    
    return True, f"Video ok: {duration:.2f}s, {size_mb:.2f}MB"


def check_intro_video(file_path: str, caption: str = "") -> Tuple[bool, str]:
    caption_lower = caption.lower()
    transcript = transcribe_first_n_seconds(file_path, TRANSCRIBE_SECONDS)
    first_words = " ".join(transcript.split()[:50])
    
    found_intro = any(k in first_words for k in INTRO_KEYWORDS)
    found_company = any(k in transcript for k in COMPANY_KEYWORDS)
    caption_intro = any(k in caption_lower for k in CAPTION_KEYWORDS)

    if caption_intro and not (found_intro or found_company):
        return False, "Caption says intro but transcript missing intro/company info"
    
    if found_intro and found_company:
        return True, "Intro video detected (self intro + company info)"
    if found_intro:
        return True, "Intro video detected (self intro)"
    if found_company:
        return True, "Intro video detected (company-focused)"
    
    return False, "No intro or company keywords in transcript"


def check_real_or_ai(file_path: str) -> Tuple[bool, str]:
    try:
        reader = imageio.get_reader(file_path, format="ffmpeg")
    except Exception as e:
        return False, f"Cannot open video: {e}"

    nframes = reader.count_frames()
    if nframes == 0:
        reader.close()
        return False, "Video has no frames"

    indices = sample_frame_indices(nframes, MAX_SAMPLED_FRAMES)

    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(encode_faces, reader.get_data(idx)): idx for idx in indices}
        for future in as_completed(futures):
            try:
                encs = future.result()
                if encs:
                    embeddings.append(encs[0])
            except:
                continue

    reader.close()

    if not embeddings:
        return False, "No face detected in sampled frames"

    mean_emb = np.mean(np.stack(embeddings), axis=0)
    consistent_count = sum(np.linalg.norm(mean_emb - emb) <= FACE_TOLERANCE for emb in embeddings)
    consistency_ratio = consistent_count / len(embeddings)

    if consistency_ratio >= 0.8:
        return True, f"Face consistent ({consistent_count}/{len(embeddings)} frames matched)"
    else:
        return False, f"Face inconsistent ({consistent_count}/{len(embeddings)} frames matched)"

def main_pipeline(file_path: str, caption: str = "") -> Dict[str, Any]:
    logs = []

   
    ok, msg = check_video_info(file_path)
    logs.append(("Video Info", ok, msg))
    if not ok:
        return {"verdict": "❌", "message": msg, "logs": logs}

    for func, name in [(check_intro_video, "Intro Check"),
                       (check_real_or_ai, "Face Consistency")]:
        try:
            result = func(file_path, caption) if func == check_intro_video else func(file_path)
            ok, msg = result
        except Exception as e:
            ok, msg = False, f"{name} failed: {e}"
        logs.append((name, ok, msg))
        if not ok:
            return {"verdict": "❌", "message": msg, "logs": logs}

    return {"verdict": "✅", "message": "Video Accepted", "logs": logs}


def print_result(res: Dict[str, Any]):
    print("\n=== FINAL VERDICT ===")
    print(res.get("verdict", "?"), res.get("message", ""))
    print("\n=== DETAILS ===")
    for step, ok, msg in res.get("logs", []):
        print(f"{step}: {'✅' if ok else '❌'} - {msg}")

if __name__ == "__main__":
    file_path = "ssvid.net---MBA-Finance-interview-for-MNC-companies-svprofessionals_480p.mp4"
    caption = "My intro video"
    result = main_pipeline(file_path, caption)
    print_result(result)
