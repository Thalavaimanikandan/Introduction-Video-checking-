import cv2
import os

def check_intro_video(video_path, caption_text="", max_duration=30, debug=False):
    text_to_check = (caption_text + " " + os.path.basename(video_path)).lower()
    caption_claim = any(word in text_to_check for word in ["intro", "introduction", "opening"])
    if debug:
        print(f"[DEBUG] Text Checked: '{text_to_check}' -> Caption Claim = {caption_claim}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Cannot open video file ❌"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = frame_count / fps
    cap.release()

    if debug:
        print(f"[DEBUG] Video Duration = {duration:.2f}s (Max allowed = {max_duration}s)")

    if caption_claim and duration <= max_duration:
        result = "Real Intro Video ✅"
    elif caption_claim:
        result = "Fake Intro Video ⚠️"
    else:
        result = "Not an Intro Video ❌"

    if debug:
        print(f"[DEBUG] Final Result -> {result}")
    return result

print(check_intro_video("Asked AI to make Mr Beast more Handsome and slightly cartoonish.mp4", debug=True))
print(check_intro_video("How To Introduce Yourself_  Upsc interview #introduction.mp4", debug=True))
print(check_intro_video("Integrating Python with OpenAI Chatgpt {தமிழ்} (1).mp4", "Self Intro Video", debug=True))

