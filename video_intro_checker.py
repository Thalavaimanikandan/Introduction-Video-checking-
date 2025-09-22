from transformers import pipeline
import cv2
import os

def is_intro_caption(caption_text):
   
    keywords = ["intro video", "introduction", "opening video"]
    caption_text = caption_text.lower()
    return any(keyword in caption_text for keyword in keywords)

def analyze_video(video_path):
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps

    
    if duration > 30:  
        cap.release()
        return False
    cap.release()
   
    return True

def check_intro_video(video_path, caption_text):
    caption_claim = is_intro_caption(caption_text)
    video_valid = analyze_video(video_path)

    if caption_claim and video_valid:
        return "Real Intro Video ✅"
    elif caption_claim and not video_valid:
        return "Fake Intro Video ⚠️"
    else:
        return "Not an Intro Video ❌"



video_path = "How To Introduce Yourself_  Upsc interview #introduction.mp4"
caption_text = "My awesome intro video"

result = check_intro_video(video_path, caption_text)
print(result)
