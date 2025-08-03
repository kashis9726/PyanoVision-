import cv2
import mediapipe as mp
import numpy as np
from scipy.io.wavfile import write as write_wav
import math
import pyaudio
import wave
import os
import time
import traceback

# --- Configuration ---
# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Audio notes directory (will be created if it doesn't exist)
AUDIO_NOTES_DIR = "piano_notes"

# Define piano keys and their associated audio files (names only)
PIANO_KEYS = [
    {"note": "C4", "file": "C4.wav"},
    {"note": "D4", "file": "D4.wav"},
    {"note": "E4", "file": "E4.wav"},
    {"note": "F4", "file": "F4.wav"},
    {"note": "G4", "file": "G4.wav"},
    {"note": "A4", "file": "A4.wav"},
    {"note": "B4", "file": "B4.wav"},
    {"note": "C5", "file": "C5.wav"},
]

# Keyboard drawing parameters
KEY_WIDTH_RATIO = 0.1 # Each key takes 10% of screen width
KEY_HEIGHT = 150 # Height of the keys in pixels
KEYBOARD_Y_OFFSET = 50 # Distance from bottom of screen to top of keys

# Gesture detection parameters
TAP_THRESHOLD_Y = 15 # Vertical movement (pixels) to trigger a "tap"
HOLD_THRESHOLD_X = 20 # Horizontal tolerance for a finger to be "over" a key
FINGER_TO_TRACK = mp_hands.HandLandmark.INDEX_FINGER_TIP # Use the index finger tip for playing

# --- Audio Generation Parameters ---
SAMPLE_RATE = 44100  # samples per second (standard audio quality)
DURATION = 0.5     # seconds (duration of each note)
AMPLITUDE = 0.5    # Amplitude of the sine wave (0.0 to 1.0)

# Define frequencies for the piano notes (in Hz)
NOTE_FREQUENCIES = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
}

# Global state for piano interaction
audio_players = {}
p_audio_instance = None
last_finger_y_pos = {}
key_is_currently_pressed = {}

# --- Function to Generate Missing Audio Files ---
def ensure_audio_files_exist(output_dir, notes_frequencies, sample_rate, duration, amplitude):
    print("Checking for required audio notes...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    all_exist = True
    for note, freq in notes_frequencies.items():
        file_path = os.path.join(output_dir, f"{note}.wav")
        if not os.path.exists(file_path):
            all_exist = False
            print(f"  {note}.wav not found. Generating...")
            
            t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
            data = amplitude * np.sin(2 * np.pi * freq * t)
            
            fade_out_duration = 0.05 # seconds
            fade_out_samples = int(SAMPLE_rate * fade_out_duration)
            if fade_out_samples > 0:
                fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
                data[-fade_out_samples:] *= fade_out_curve
            
            audio_data = (data * 32767).astype(np.int16)
            
            write_wav(file_path, sample_rate, audio_data)
            print(f"  Generated {file_path}")
        else:
            print(f"  {note}.wav already exists.")
    
    if all_exist:
        print("All required audio files found.")
    else:
        print("Generated missing audio files.")


# --- Audio Loading Function ---
def load_audio_files():
    global p_audio_instance, audio_players

    if not os.path.exists(AUDIO_NOTES_DIR):
        print(f"Error: Audio notes directory '{AUDIO_NOTES_DIR}' is still missing after generation attempt.")
        return False

    try:
        p_audio_instance = pyaudio.PyAudio()
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        print("Please ensure PyAudio is correctly installed and your audio devices are working.")
        return False

    all_loaded = True
    for key_info in PIANO_KEYS:
        note = key_info["note"]
        file_name = key_info["file"]
        file_path = os.path.join(AUDIO_NOTES_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: Audio file '{file_path}' not found for note {note}. This note will not play.")
            all_loaded = False
            continue

        try:
            wf = wave.open(file_path, 'rb')
            stream = p_audio_instance.open(format=p_audio_instance.get_format_from_width(wf.getsampwidth()),
                                            channels=wf.getnchannels(),
                                            rate=wf.getframerate(),
                                            output=True)
            audio_players[note] = {"stream": stream, "wf": wf, "data": wf.readframes(wf.getnframes())}
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}. This note will not play.")
            all_loaded = False
            continue
    
    if not all_loaded:
        print("Some audio files failed to load. Virtual piano functionality might be limited.")
    if not audio_players:
        print("No audio notes were successfully loaded. Virtual piano cannot play any sound.")
        return False
    return True

# --- Play Audio Function ---
def play_note(note):
    print(f"DEBUG: Attempting to play note: {note}") # Added DEBUG print
    if note in audio_players:
        player = audio_players[note]
        try:
            print(f"DEBUG: Rewinding waveform for {note}") # Added DEBUG print
            player["wf"].rewind() # Corrected: 'rewindframe()' to 'rewind()'
            
            # This is the simplified, more robust way to play short notes:
            # Just write the data. PyAudio handles buffering.
            print(f"DEBUG: Writing audio data for {note} to stream.") # Added DEBUG print
            player["stream"].write(player["data"])
            print(f"DEBUG: Successfully wrote audio data for {note}.") # Added DEBUG print

        except Exception as e:
            print(f"ERROR: Exception caught during playing note {note}: {e}")
            traceback.print_exc() # Print full traceback for this specific error

# --- Main Logic ---
def main():
    global p_audio_instance, last_finger_y_pos, key_is_currently_pressed

    print("DEBUG: Starting main function.")

    # 1. Ensure audio files exist (generate if missing)
    print("DEBUG: Checking/Generating audio files...")
    ensure_audio_files_exist(
        AUDIO_NOTES_DIR,
        NOTE_FREQUENCIES,
        SAMPLE_RATE,
        DURATION,
        AMPLITUDE
    )
    print("DEBUG: Audio file check/generation complete.")

    # 2. Load audio files into PyAudio
    print("DEBUG: Loading audio into PyAudio...")
    if not load_audio_files():
        print("DEBUG: Failed to load audio files. Exiting.")
        print("Exiting due to critical audio loading issues. Check 'piano_notes' folder and WAV files.")
        return
    print("DEBUG: Audio files loaded successfully.")

    # 3. Webcam capture
    print("DEBUG: Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("DEBUG: Failed to open webcam. Exiting.")
        print("Error: Could not open video stream. Please ensure webcam is connected and not in use by other applications.")
        print("You might also try changing 'cv2.VideoCapture(0)' to 'cv2.VideoCapture(1)' or '2'.")
        return
    print("DEBUG: Webcam opened successfully.")

    # Set up OpenCV window for fullscreen
    cv2.namedWindow('Virtual Piano', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Virtual Piano', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("DEBUG: Window set to fullscreen.")

    # Get initial frame dimensions for drawing keys (these will be fullscreen dimensions)
    ret, frame = cap.read()
    if not ret:
        print("DEBUG: Failed to read initial frame from webcam. Exiting.")
        print("Failed to read initial frame from webcam. Check webcam connection.")
        return
    print("DEBUG: Initial frame read.")
    
    frame_h, frame_w, _ = frame.shape
    
    # 4. Calculate and store key positions dynamically (for right side)
    print("DEBUG: Calculating key positions...")
    num_keys = len(PIANO_KEYS)
    key_width = int(frame_w * KEY_WIDTH_RATIO)
    
    total_keys_width = key_width * num_keys
    # Adjust key_width to perfectly fit if it overflows, for responsiveness
    if total_keys_width > frame_w * 0.4: # Limit to about 40% of screen width if too wide
        key_width = int(frame_w * 0.4 / num_keys)
        total_keys_width = key_width * num_keys

    # Position on the right side: (frame_width - total_keys_width - margin_from_right)
    margin_from_right = 20 # Pixels from the right edge
    keyboard_start_x = frame_w - total_keys_width - margin_from_right
    keyboard_start_y = frame_h - KEY_HEIGHT - KEYBOARD_Y_OFFSET
    
    key_rects = {}
    for i, key_info in enumerate(PIANO_KEYS):
        x1 = keyboard_start_x + i * key_width
        y1 = keyboard_start_y
        x2 = x1 + key_width
        y2 = y1 + KEY_HEIGHT
        key_rects[key_info["note"]] = (x1, y1, x2, y2)
        key_is_currently_pressed[key_info["note"]] = False
    print("DEBUG: Key positions calculated.")

    print("Virtual Piano ready. Position your hand over the keys. Press 'q' to quit.")

    # 5. Main loop for video processing and interaction
    print("DEBUG: Entering main video loop. Looking for 'q' to quit.")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting video loop.")
                break

            frame = cv2.flip(frame, 1) # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw virtual piano keys on the frame
            for i, key_info in enumerate(PIANO_KEYS):
                note = key_info["note"]
                x1, y1, x2, y2 = key_rects[note]
                
                color = (0, 255, 255) if key_is_currently_pressed[note] else (200, 200, 200) 

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                text_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x1 + (key_width - text_size[0]) // 2
                text_y = y1 + (KEY_HEIGHT + text_size[1]) // 2
                cv2.putText(frame, note, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

                    finger_tip_lm = hand_landmarks.landmark[FINGER_TO_TRACK]
                    finger_tip_x = int(finger_tip_lm.x * frame_w)
                    finger_tip_y = int(finger_tip_lm.y * frame_h)

                    cv2.circle(frame, (finger_tip_x, finger_tip_y), 10, (255, 0, 0), -1)

                    # Reset all key active statuses for the current frame initially
                    for note in key_is_currently_pressed:
                        key_is_currently_pressed[note] = False

                    current_key_note_hovered = None
                    for key_info in PIANO_KEYS:
                        note = key_info["note"]
                        x1, y1, x2, y2 = key_rects[note]
                        
                        if (x1 - HOLD_THRESHOLD_X < finger_tip_x < x2 + HOLD_THRESHOLD_X and
                            y1 - KEY_HEIGHT // 2 < finger_tip_y < y2 + KEY_HEIGHT // 2):
                            current_key_note_hovered = note
                            break

                    if current_key_note_hovered:
                        if FINGER_TO_TRACK not in last_finger_y_pos:
                            last_finger_y_pos[FINGER_TO_TRACK] = finger_tip_y
                            
                        if (finger_tip_y - last_finger_y_pos[FINGER_TO_TRACK] > TAP_THRESHOLD_Y) and \
                           (keyboard_start_y <= finger_tip_y <= keyboard_start_y + KEY_HEIGHT):
                            if not key_is_currently_pressed[current_key_note_hovered]: 
                                play_note(current_key_note_hovered)
                                key_is_currently_pressed[current_key_note_hovered] = True
                        else:
                            if key_is_currently_pressed[current_key_note_hovered]:
                                key_is_currently_pressed[current_key_note_hovered] = False

                        last_finger_y_pos[FINGER_TO_TRACK] = finger_tip_y 
                    else:
                        if FINGER_TO_TRACK in last_finger_y_pos:
                            del last_finger_y_pos[FINGER_TO_TRACK]
                        for note in key_is_currently_pressed:
                            key_is_currently_pressed[note] = False


            cv2.imshow('Virtual Piano', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\nAn unexpected error occurred during the main loop: {e}")
        traceback.print_exc()

    finally:
        print("\nReleasing resources...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        hands.close()
        for note in audio_players:
            if audio_players[note]["stream"].is_active():
                audio_players[note]["stream"].stop_stream()
            audio_players[note]["stream"].close()
            audio_players[note]["wf"].close()
        if p_audio_instance:
            p_audio_instance.terminate()
        print("Application closed.")

# Entry point for the script
if __name__ == "__main__":
    main()