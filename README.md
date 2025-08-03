PyanoVision: A Virtual Hand-Controlled Piano
🎹 Project Overview
Welcome to PyanoVision! This is my mini-project that turns your webcam into a virtual piano. Using computer vision and hand tracking, you can play musical notes just by "tapping" on a virtual keyboard with your fingers. It's a fun and engaging way to combine technology with creativity.

I built this project to learn about real-time hand gesture recognition and audio feedback. It's a great example of how you can use Python libraries like OpenCV and MediaPipe to create interactive applications.

✨ Features
Hand Tracking with MediaPipe: The app uses MediaPipe to accurately detect and track your hand's position in real time.

Virtual Keyboard: A digital piano keyboard is displayed right on your webcam feed.

Gesture-Based Playing: A simple downward "tapping" motion with your index finger is all it takes to play a note.

Instant Audio Feedback: The app plays the corresponding piano note as soon as it detects a tap.

Dynamic Visuals: The virtual piano keys light up when you play them, giving you visual feedback.

Automatic Note Generation: The first time you run the app, it automatically creates the necessary audio files for each note. No need to download anything extra!

🚀 How to Run It
What you need:
Python 3.x

A webcam

Installation:
Clone this repository:

Bash


cd PyanoVision
Install the required libraries:

Bash

pip install opencv-python mediapipe numpy scipy pyaudio
Let's Play!
Run the script from your terminal:

Bash

python piano_cv_project.py
(Note: If you're on a Mac, you might need to run this from your main terminal, not inside an IDE like VS Code, due to PyAudio permissions.)

Webcam View: A full-screen window will open, showing your webcam feed. On the right side of the screen, you'll see a virtual piano keyboard.

Play a Note: Hold your hand up and move your index finger in a downward "tapping" motion over a key. The key will light up, and you'll hear the note play!

Exit: To quit the application, just press the q key on your keyboard.

🛠️ My Technical Stack
Python: The main programming language for the project.

OpenCV: Used to capture video from the webcam and to draw the virtual keyboard on the screen.

MediaPipe: The key to hand tracking. It gives me the precise coordinates of my finger in real time.

NumPy & SciPy: Used to programmatically generate the audio files for the piano notes. This was a really cool part of the project!

PyAudio: A library that lets Python play audio directly to your speakers.

This project was a great learning experience. 
