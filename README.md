🎥 Video Verification & Intro Checker

      This project is a video analysis pipeline that verifies if a given video is a valid self-introduction video and checks whether the video is real or AI-generated.
      
      It combines speech transcription (Whisper) + face consistency detection (dlib) to ensure authenticity.
      
✨ Features

    ✅ Check video properties (size, duration).
    
    ✅ Transcribe first 25 seconds of audio using OpenAI Whisper.
    
    ✅ Detect introductory keywords (e.g., "my name", "I am", "company").
    
    ✅ Validate if caption matches transcript content.
    
    ✅ Perform face consistency check using dlib embeddings (detects deepfakes).
    
    ✅ Final verdict: Accept ✅ or Reject ❌ video with logs.

🛠️ Requirements

      Python 3.9+
      
      ffmpeg
       (must be installed & available in PATH)
      
      GPU with CUDA (optional, for faster Whisper inference)

📦 Installation:
    1.Clone this repo
        git clone https://github.com/yourname/video-intro-checker.git
        cd video-intro-checker

    2.Create a virtual environment
          python -m venv venv
          source venv/bin/activate

    3.Install dependencies
         pip install -r requirements.txt


📂 Model Files

You need to download dlib pretrained models:

Face landmarks predictor
shape_predictor_68_face_landmarks.dat

Face recognition model
dlib_face_recognition_resnet_model_v1.dat

Place them in the project root.


▶️ Usage

Update the video path in main.py:

if __name__ == "__main__":
    file_path = "your_video.mp4"
    caption = "My intro video"
    result = main_pipeline(file_path, caption)
    print_result(result)


Run:

python main.py

📊 Example Output
=== FINAL VERDICT ===
✅ Video Accepted

=== DETAILS ===
Video Info: ✅ - Video ok: 58.23s, 12.45MB
Intro Check: ✅ - Intro video detected (self intro + company info)
Face Consistency: ✅ - Face consistent (18/20 frames matched)


If the video fails checks:

=== FINAL VERDICT ===
❌ Caption says intro but transcript missing intro/company info

⚡ Customization

Change transcription length:
TRANSCRIBE_SECONDS = 25

Adjust face match tolerance:
FACE_TOLERANCE = 0.6

Update keywords inside:

INTRO_KEYWORDS

COMPANY_KEYWORDS

CAPTION_KEYWORDS

📌 Notes

Whisper model (base) is used by default. You can switch to small, medium, or large for higher accuracy:

WHISPER_MODEL = "small"


First run may take time (Whisper downloads the model).

Works best on clear speech and frontal face videos.

      
