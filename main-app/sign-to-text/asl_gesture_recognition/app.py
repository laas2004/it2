# from flask import Flask, render_template, request, jsonify
# import sys
# import os

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/detect', methods=['GET', 'POST'])
# def detect_sign():
#     try:
#         # This will depend on how your main.py works
#         # Call your existing sign detection logic here
#         # Example:
#         # from main import detect_sign_language
#         # result = detect_sign_language()
        
#         # For now, return mock data
#         detected_text = "Hello world"  # Replace with actual detection
#         confidence = 0.85
        
#         return jsonify({
#             'success': True,
#             'detected_text': detected_text,
#             'confidence': confidence
#         })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     app.run(port=5002, debug=True)


# v1
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# # Make sure punkt is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",           # American Sign Language
#     "bsl": "signs_bsl",       # British Sign Language  
#     "isl": "signs_isl"        # Indian Sign Language
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()

# # -------------------- Text to Sign Functions --------------------

# def get_signs_path(language):
#     """Get the path for the selected sign language."""
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     """List all available signs for the selected language."""
#     signs_path = get_signs_path(language)
#     print(f"🔍 Looking for signs in: {signs_path} for language: {language}")
    
#     if not os.path.exists(signs_path):
#         print(f"❌ Signs directory does not exist: {signs_path}")
#         return []
    
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     print(f"📹 Found {len(vid_names)} sign videos for {language}: {vid_names}")
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     """Find the closest match for a word or phrase in the signs database."""
#     # Try to find a direct match for the phrase
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
    
#     if phrase_match in available_signs:
#         return phrase_match
    
#     # If no direct match, fallback to individual word matching
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
    
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     signs_path = get_signs_path(language)
    
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 # Resize to consistent size
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#                 print(f"✅ Loaded: {sign} for {language}")
#             except Exception as e:
#                 print(f"❌ Error loading clip {sign_path}: {e}")
#                 return False
#         else:
#             print(f"⚠️ Missing file: {sign_path}")
#             return False

#     if clips:
#         try:
#             print(f"🎞️ Merging {len(clips)} clips for {language}...")
#             final_clip = concatenate_videoclips(clips, method="compose")
#             # Ensure output directory exists
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             # Write the video file
#             final_clip.write_videofile(
#                 output_path, 
#                 codec="libx264", 
#                 audio=False,
#                 fps=24
#             )
#             # Close clips to free memory
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             print(f"✅ Output saved to: {output_path}")
#             return True
#         except Exception as e:
#             print(f"❌ Error merging clips: {e}")
#             return False
#     else:
#         print("❌ No clips to merge.")
#         return False

# def text_to_sign(text, language):
#     """Main function to convert text to sign language video"""
#     print(f"🎯 Starting translation for: '{text}' in {language}")
    
#     # Sanitize the input
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     print(f"🧹 Sanitized Input: '{sanitized_text}'")

#     # Check for direct phrase matches first
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you", 
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
    
#     if sanitized_text in phrase_mappings:
#         phrase_sign = phrase_mappings[sanitized_text]
#         print(f"✅ Found phrase mapping: '{sanitized_text}' -> '{phrase_sign}'")
#         success = merge_signs([phrase_sign], language)
#         return success
    
#     # Word-by-word processing for other text
#     words = sanitized_text.split()
#     final_sequence = []

#     # Process each word
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             print(f"✅ '{w}' found in database as '{db_match}'")
#             final_sequence.append(db_match)
#         else:
#             print(f"❌ '{w}' not found in database, trying to spell it...")
#             spelled = spell_word(w, language)
#             if spelled:
#                 print(f"🔤 '{w}' → spelling as: {spelled}")
#                 final_sequence.extend(spelled)
#             else:
#                 print(f"❌ Cannot represent '{w}', skipping...")

#     print(f"📋 Final sequence for {language}: {final_sequence}")
    
#     if final_sequence:
#         success = merge_signs(final_sequence, language)
#         return success
#     else:
#         print("❌ No matching or spellable words found.")
#         return False

# # -------------------- Sign to Text (Gesture Recognition) Functions --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running
    
#     try:
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer()
#         camera = cv2.VideoCapture(0)
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         is_camera_running = True
#         print("✅ Gesture recognition initialized successfully")
#         return True
#     except Exception as e:
#         print(f"❌ Failed to initialize gesture recognition: {e}")
#         return False

# def update_gesture_recognition():
#     """Update gesture recognition in a separate thread"""
#     global current_gesture, gesture_sentence, last_gesture_time, is_camera_running
    
#     while is_camera_running:
#         if camera and gesture_recognizer:
#             ret, frame = camera.read()
#             if ret:
#                 frame = cv2.flip(frame, 1)
#                 processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
#                 current_gesture = detected_gesture
                
#                 # Add to sentence logic
#                 current_time = time.time()
#                 if (detected_gesture != current_gesture or 
#                     current_time - last_gesture_time > 2.0):
                    
#                     if detected_gesture not in ["No hand detected", "Unknown"]:
#                         gesture_sentence += detected_gesture + " "
#                         last_gesture_time = current_time

# def generate_frames():
#     """Generate video frames for streaming"""
#     global camera, gesture_recognizer, is_camera_running
    
#     while is_camera_running and camera:
#         ret, frame = camera.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
        
#         # Process frame with gesture recognition
#         if gesture_recognizer:
#             processed_frame, _ = gesture_recognizer.recognize_gesture(frame)
            
#             # Add text overlay
#             height, width = processed_frame.shape[:2]
            
#             # Create overlay for text
#             overlay = processed_frame.copy()
#             cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#             cv2.rectangle(overlay, (10, height - 150), (width - 10, height - 10), (0, 0, 0), -1)
            
#             # Blend overlay with frame
#             processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
            
#             # Display current gesture
#             cv2.putText(processed_frame, f"Current Gesture: {current_gesture}", 
#                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
#             # Display sentence
#             cv2.putText(processed_frame, "Recognized Text:", 
#                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             # Word wrap for long sentences
#             sentence_words = gesture_sentence.split()
#             line_length = 40
#             lines = []
#             current_line = ""
            
#             for word in sentence_words:
#                 if len(current_line + word) <= line_length:
#                     current_line += word + " "
#                 else:
#                     lines.append(current_line.strip())
#                     current_line = word + " "
            
#             if current_line:
#                 lines.append(current_line.strip())
            
#             for i, line in enumerate(lines[-3:]):  # Show last 3 lines
#                 cv2.putText(processed_frame, line, 
#                            (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             # Display instructions
#             cv2.putText(processed_frame, "Hold gesture for 2 seconds to add to text", 
#                        (20, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
#             frame = processed_frame
        
#         # Encode frame as JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def stop_gesture_recognition():
#     """Stop gesture recognition and release resources"""
#     global is_camera_running, camera
    
#     is_camera_running = False
#     if camera:
#         camera.release()
#     cv2.destroyAllWindows()

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
            
#         print(f"📥 Received text to translate: '{text}' in {language}")
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'language': language,
#                 'message': f'Sign language video created for: {text} ({language.upper()})',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Could not generate {language.upper()} sign language video. Please try a different word.'
#             })
            
#     except Exception as e:
#         print(f"❌ Route error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def serve_video():
#     """Serve the generated text-to-sign video file"""
#     try:
#         if os.path.exists(STATIC_VIDEO_PATH):
#             return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Video not found'
#             }), 404
#     except Exception as e:
#         print(f"❌ Video serve error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# # -------------------- Gesture Recognition Routes --------------------

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     """Start the gesture recognition system"""
#     global gesture_sentence
    
#     try:
#         success = init_gesture_recognition()
#         if success:
#             # Start gesture recognition in background thread
#             thread = threading.Thread(target=update_gesture_recognition)
#             thread.daemon = True
#             thread.start()
            
#             gesture_sentence = ""  # Reset sentence
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Gesture recognition started'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to initialize camera or gesture recognition'
#             })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     """Stop the gesture recognition system"""
#     stop_gesture_recognition()
#     return jsonify({
#         'success': True,
#         'message': 'Gesture recognition stopped'
#     })

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route for gesture recognition"""
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     """Get current gesture and sentence data"""
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip()
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     """Clear the recognized sentence"""
#     global gesture_sentence
#     gesture_sentence = ""
#     return jsonify({
#         'success': True,
#         'message': 'Sentence cleared'
#     })

# @app.route('/debug')
# def debug():
#     """Debug endpoint to check paths and available signs"""
#     debug_info = {
#         'base_dir': BASE_DIR,
#         'available_languages': list(SIGN_LANGUAGES.keys()),
#         'current_working_dir': os.getcwd(),
#         'gesture_recognition_initialized': gesture_recognizer is not None,
#         'camera_running': is_camera_running
#     }
    
#     for lang in SIGN_LANGUAGES:
#         signs_path = get_signs_path(lang)
#         debug_info[f'signs_path_{lang}'] = signs_path
#         debug_info[f'signs_exists_{lang}'] = os.path.exists(signs_path)
#         debug_info[f'available_signs_{lang}'] = get_words_in_database(lang)
    
#     return jsonify(debug_info)

# @app.teardown_appcontext
# def close_camera(exception=None):
#     """Ensure camera is released when app closes"""
#     stop_gesture_recognition()

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")
    
#     for lang, folder in SIGN_LANGUAGES.items():
#         path = os.path.join(BASE_DIR, folder)
#         exists = os.path.exists(path)
#         print(f"📁 {lang.upper()} path: {path} - {'✅ Exists' if exists else '❌ Missing'}")
#         if exists:
#             signs = get_words_in_database(lang)
#             print(f"   📹 Available signs: {len(signs)}")
    
#     app.run(port=5001, debug=True)



# v2-trying to make the camera work
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# # Make sure punkt is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",           # American Sign Language
#     "bsl": "signs_bsl",       # British Sign Language  
#     "isl": "signs_isl"        # Indian Sign Language
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# gesture_recognition_available = False
# gesture_thread = None

# # Try to import gesture recognition dependencies
# try:
#     import cv2
#     print("✅ OpenCV imported successfully")
    
#     # Try to import gesture_recognition module
#     try:
#         from gesture_recognition import GestureRecognizer
#         print("✅ GestureRecognizer imported successfully")
#         gesture_recognition_available = True
#     except ImportError as e:
#         print(f"❌ Failed to import GestureRecognizer: {e}")
#     except Exception as e:
#         print(f"❌ Error importing GestureRecognizer: {e}")
        
# except ImportError as e:
#     print(f"❌ OpenCV import failed: {e}")
# except Exception as e:
#     print(f"❌ Error loading dependencies: {e}")

# # -------------------- Text to Sign Functions --------------------

# def get_signs_path(language):
#     """Get the path for the selected sign language."""
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     """List all available signs for the selected language."""
#     signs_path = get_signs_path(language)
    
#     if not os.path.exists(signs_path):
#         return []
    
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     """Find the closest match for a word or phrase in the signs database."""
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
    
#     if phrase_match in available_signs:
#         return phrase_match
    
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
    
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     signs_path = get_signs_path(language)
    
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 return False
#         else:
#             return False

#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(
#                 output_path, 
#                 codec="libx264", 
#                 audio=False,
#                 fps=24
#             )
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             return False
#     else:
#         return False

# def text_to_sign(text, language):
#     """Main function to convert text to sign language video"""
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()

#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you", 
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
    
#     if sanitized_text in phrase_mappings:
#         phrase_sign = phrase_mappings[sanitized_text]
#         success = merge_signs([phrase_sign], language)
#         return success
    
#     words = sanitized_text.split()
#     final_sequence = []

#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)

#     if final_sequence:
#         success = merge_signs(final_sequence, language)
#         return success
#     else:
#         return False

# # -------------------- Sign to Text (Gesture Recognition) Functions --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running
    
#     if not gesture_recognition_available:
#         return False
        
#     try:
#         print("🔄 Initializing GestureRecognizer...")
#         gesture_recognizer = GestureRecognizer()
#         print("✅ GestureRecognizer initialized")
        
#         print("🔄 Initializing camera...")
#         camera = cv2.VideoCapture(0)
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         is_camera_running = True
#         print("✅ Camera initialized successfully")
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize gesture recognition: {e}")
#         return False

# def update_gesture_recognition():
#     """Update gesture recognition in a separate thread"""
#     global current_gesture, gesture_sentence, last_gesture_time, is_camera_running
    
#     print("🔄 Starting gesture recognition thread...")
    
#     while is_camera_running and gesture_recognition_available:
#         try:
#             if camera and gesture_recognizer and is_camera_running:
#                 ret, frame = camera.read()
#                 if ret:
#                     frame = cv2.flip(frame, 1)
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
#                     current_gesture = detected_gesture
                    
#                     current_time = time.time()
#                     if (detected_gesture != current_gesture or 
#                         current_time - last_gesture_time > 2.0):
                        
#                         if detected_gesture not in ["No hand detected", "Unknown"]:
#                             gesture_sentence += detected_gesture + " "
#                             last_gesture_time = current_time
#                             print(f"📝 Added to sentence: {detected_gesture}")
#                 else:
#                     time.sleep(0.1)
#             else:
#                 time.sleep(0.1)
#         except Exception as e:
#             print(f"❌ Error in gesture recognition thread: {e}")
#             time.sleep(0.1)

# def generate_frames():
#     """Generate video frames for streaming"""
#     global camera, gesture_recognizer, is_camera_running
    
#     print("🎥 Starting video stream...")
    
#     while is_camera_running and camera:
#         try:
#             ret, frame = camera.read()
#             if not ret:
#                 break
                
#             frame = cv2.flip(frame, 1)
            
#             # Process frame with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, _ = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Add text overlay
#                     height, width = processed_frame.shape[:2]
                    
#                     # Create overlay for text
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 150), (width - 10, height - 10), (0, 0, 0), -1)
                    
#                     # Blend overlay with frame
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Display current gesture
#                     cv2.putText(processed_frame, f"Current Gesture: {current_gesture}", 
#                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
#                     # Display sentence
#                     cv2.putText(processed_frame, "Recognized Text:", 
#                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
#                     # Word wrap for long sentences
#                     sentence_words = gesture_sentence.split()
#                     line_length = 40
#                     lines = []
#                     current_line = ""
                    
#                     for word in sentence_words:
#                         if len(current_line + word) <= line_length:
#                             current_line += word + " "
#                         else:
#                             lines.append(current_line.strip())
#                             current_line = word + " "
                    
#                     if current_line:
#                         lines.append(current_line.strip())
                    
#                     for i, line in enumerate(lines[-3:]):
#                         cv2.putText(processed_frame, line, 
#                                    (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
#                     frame = processed_frame
#                 except Exception as e:
#                     print(f"❌ Error processing frame: {e}")
            
#             # Encode frame as JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
#         except Exception as e:
#             print(f"❌ Error in video stream: {e}")
#             break

# def stop_gesture_recognition():
#     """Stop gesture recognition and release resources"""
#     global is_camera_running, camera, gesture_thread
    
#     print("🛑 Stopping gesture recognition...")
#     is_camera_running = False
    
#     # Wait a bit for threads to stop
#     time.sleep(1)
    
#     if camera:
#         camera.release()
#         camera = None
#         print("✅ Camera released")
    
#     gesture_thread = None

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html', gesture_available=gesture_recognition_available)

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
            
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'language': language,
#                 'message': f'Sign language video created for: {text} ({language.upper()})',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Could not generate {language.upper()} sign language video. Please try a different word.'
#             })
            
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def serve_video():
#     """Serve the generated text-to-sign video file"""
#     try:
#         if os.path.exists(STATIC_VIDEO_PATH):
#             return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Video not found'
#             }), 404
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# # -------------------- Gesture Recognition Routes --------------------

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     """Start the gesture recognition system"""
#     global gesture_sentence, gesture_thread, is_camera_running
    
#     print("🎯 Starting gesture recognition...")
    
#     if not gesture_recognition_available:
#         return jsonify({
#             'success': False,
#             'error': 'Gesture recognition is unavailable.'
#         })
    
#     # Stop any existing camera session
#     if is_camera_running:
#         stop_gesture_recognition()
#         time.sleep(1)
    
#     try:
#         success = init_gesture_recognition()
#         if success:
#             # Reset sentence
#             gesture_sentence = ""
            
#             # Start gesture recognition in background thread
#             gesture_thread = threading.Thread(target=update_gesture_recognition)
#             gesture_thread.daemon = True
#             gesture_thread.start()
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Gesture recognition started successfully'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to initialize camera. Please check if camera is available.'
#             })
#     except Exception as e:
#         print(f"❌ Error starting gesture recognition: {e}")
#         return jsonify({
#             'success': False,
#             'error': f'Error starting gesture recognition: {str(e)}'
#         })

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     """Stop the gesture recognition system"""
#     stop_gesture_recognition()
#     return jsonify({
#         'success': True,
#         'message': 'Gesture recognition stopped'
#     })

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route for gesture recognition"""
#     if not is_camera_running:
#         return "Camera not running", 400
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     """Get current gesture and sentence data"""
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'gesture_available': gesture_recognition_available,
#         'camera_running': is_camera_running
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     """Clear the recognized sentence"""
#     global gesture_sentence
#     gesture_sentence = ""
#     return jsonify({
#         'success': True,
#         'message': 'Sentence cleared'
#     })

# @app.route('/debug')
# def debug():
#     """Debug endpoint to check paths and available signs"""
#     debug_info = {
#         'base_dir': BASE_DIR,
#         'available_languages': list(SIGN_LANGUAGES.keys()),
#         'current_working_dir': os.getcwd(),
#         'gesture_recognition_available': gesture_recognition_available,
#         'gesture_recognition_initialized': gesture_recognizer is not None,
#         'camera_running': is_camera_running,
#         'current_gesture': current_gesture,
#         'gesture_sentence': gesture_sentence
#     }
    
#     for lang in SIGN_LANGUAGES:
#         signs_path = get_signs_path(lang)
#         debug_info[f'signs_path_{lang}'] = signs_path
#         debug_info[f'signs_exists_{lang}'] = os.path.exists(signs_path)
#         debug_info[f'available_signs_{lang}'] = get_words_in_database(lang)
    
#     return jsonify(debug_info)

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")
#     print(f"🎯 Gesture recognition available: {gesture_recognition_available}")
    
#     app.run(port=5001, debug=True)

# # v3-trying old code
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# # Make sure punkt is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",           # American Sign Language
#     "bsl": "signs_bsl",       # British Sign Language  
#     "isl": "signs_isl"        # Indian Sign Language
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No hand detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# gesture_recognition_available = False
# gesture_thread = None

# # Try to import gesture recognition dependencies
# try:
#     import cv2
#     print("✅ OpenCV imported successfully")
    
#     # Try to import gesture_recognition module
#     try:
#         from gesture_recognition import GestureRecognizer
#         print("✅ GestureRecognizer imported successfully")
#         gesture_recognition_available = True
#     except ImportError as e:
#         print(f"❌ Failed to import GestureRecognizer: {e}")
#     except Exception as e:
#         print(f"❌ Error importing GestureRecognizer: {e}")
        
# except ImportError as e:
#     print(f"❌ OpenCV import failed: {e}")
# except Exception as e:
#     print(f"❌ Error loading dependencies: {e}")

# # -------------------- Text to Sign Functions --------------------

# def get_signs_path(language):
#     """Get the path for the selected sign language."""
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     """List all available signs for the selected language."""
#     signs_path = get_signs_path(language)
    
#     if not os.path.exists(signs_path):
#         return []
    
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     """Find the closest match for a word or phrase in the signs database."""
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
    
#     if phrase_match in available_signs:
#         return phrase_match
    
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
    
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     signs_path = get_signs_path(language)
    
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 return False
#         else:
#             return False

#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(
#                 output_path, 
#                 codec="libx264", 
#                 audio=False,
#                 fps=24
#             )
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             return False
#     else:
#         return False

# def text_to_sign(text, language):
#     """Main function to convert text to sign language video"""
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()

#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you", 
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
    
#     if sanitized_text in phrase_mappings:
#         phrase_sign = phrase_mappings[sanitized_text]
#         success = merge_signs([phrase_sign], language)
#         return success
    
#     words = sanitized_text.split()
#     final_sequence = []

#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)

#     if final_sequence:
#         success = merge_signs(final_sequence, language)
#         return success
#     else:
#         return False

# # -------------------- Sign to Text (Gesture Recognition) Functions --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running, gesture_recognition_available
    
#     if not gesture_recognition_available:
#         return False
        
#     try:
#         print("🔄 Initializing GestureRecognizer...")
#         gesture_recognizer = GestureRecognizer()
#         if not gesture_recognizer.is_model_loaded():
#             print("❌ Gesture recognizer model not loaded")
#             return False
            
#         print("🔄 Initializing camera...")
#         camera = cv2.VideoCapture(0)
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         is_camera_running = True
#         print("✅ Camera initialized successfully")
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize gesture recognition: {e}")
#         return False

# def update_gesture_recognition():
#     """Update gesture recognition in a separate thread"""
#     global current_gesture, gesture_sentence, last_gesture_time, is_camera_running
    
#     print("🔄 Starting gesture recognition thread...")
    
#     while is_camera_running and gesture_recognition_available:
#         try:
#             if camera and gesture_recognizer and is_camera_running:
#                 ret, frame = camera.read()
#                 if ret:
#                     frame = cv2.flip(frame, 1)
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
#                     current_gesture = detected_gesture
                    
#                     # Add to sentence logic - only add valid gestures
#                     current_time = time.time()
#                     if (detected_gesture not in ["No hand detected", "Unknown"] and 
#                         current_time - last_gesture_time > 2.0):
                        
#                         gesture_sentence += detected_gesture + " "
#                         last_gesture_time = current_time
#                         print(f"📝 Added to sentence: {detected_gesture}")
#                 else:
#                     time.sleep(0.1)
#             else:
#                 time.sleep(0.1)
#         except Exception as e:
#             print(f"❌ Error in gesture recognition thread: {e}")
#             time.sleep(0.1)

# def generate_frames():
#     """Generate video frames for streaming"""
#     global camera, is_camera_running
    
#     print("🎥 Starting video stream...")
    
#     frame_skip = 2  # Process every 3rd frame for performance
#     frame_counter = 0
    
#     while is_camera_running and camera:
#         try:
#             ret, frame = camera.read()
#             if not ret:
#                 break
                
#             frame = cv2.flip(frame, 1)
#             frame_counter += 1
            
#             # Create a clean frame for display (without gesture recognition overlay)
#             display_frame = frame.copy()
            
#             # Only process every 3rd frame for performance
#             if frame_counter % frame_skip == 0 and gesture_recognizer:
#                 processed_frame, _ = gesture_recognizer.recognize_gesture(frame)
#                 # We get the gesture info but don't use the processed_frame for display
#                 # to avoid overlapping text
            
#             # Add our own clean overlay (no overlapping text)
#             height, width = display_frame.shape[:2]
            
#             # Create semi-transparent overlay for text
#             overlay = display_frame.copy()
#             cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#             cv2.rectangle(overlay, (10, height - 150), (width - 10, height - 10), (0, 0, 0), -1)
            
#             # Blend overlay with frame
#             display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            
#             # Display current gesture (clean, no overlap)
#             cv2.putText(display_frame, f"Current Gesture: {current_gesture}", 
#                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
#             # Display sentence
#             cv2.putText(display_frame, "Recognized Text:", 
#                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             # Word wrap for long sentences
#             sentence_words = gesture_sentence.split()
#             line_length = 40
#             lines = []
#             current_line = ""
            
#             for word in sentence_words:
#                 if len(current_line + word) <= line_length:
#                     current_line += word + " "
#                 else:
#                     lines.append(current_line.strip())
#                     current_line = word + " "
            
#             if current_line:
#                 lines.append(current_line.strip())
            
#             for i, line in enumerate(lines[-3:]):
#                 cv2.putText(display_frame, line, 
#                            (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             # Display instructions
#             cv2.putText(display_frame, "Hold gesture for 2 seconds to add to text", 
#                        (20, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
#             # Encode frame as JPEG
#             ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
#             frame_bytes = buffer.tobytes()
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
#         except Exception as e:
#             print(f"❌ Error in video stream: {e}")
#             break

# def stop_gesture_recognition():
#     """Stop gesture recognition and release resources"""
#     global is_camera_running, camera, gesture_thread
    
#     print("🛑 Stopping gesture recognition...")
#     is_camera_running = False
    
#     # Wait a bit for threads to stop
#     time.sleep(1)
    
#     if camera:
#         camera.release()
#         camera = None
#         print("✅ Camera released")
    
#     gesture_thread = None

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html', gesture_available=gesture_recognition_available)

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
            
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'language': language,
#                 'message': f'Sign language video created for: {text} ({language.upper()})',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Could not generate {language.upper()} sign language video. Please try a different word.'
#             })
            
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def serve_video():
#     """Serve the generated text-to-sign video file"""
#     try:
#         if os.path.exists(STATIC_VIDEO_PATH):
#             return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Video not found'
#             }), 404
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# # -------------------- Gesture Recognition Routes --------------------

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     """Start the gesture recognition system"""
#     global gesture_sentence, gesture_thread, is_camera_running
    
#     print("🎯 Starting gesture recognition...")
    
#     if not gesture_recognition_available:
#         return jsonify({
#             'success': False,
#             'error': 'Gesture recognition is unavailable.'
#         })
    
#     # Stop any existing camera session
#     if is_camera_running:
#         stop_gesture_recognition()
#         time.sleep(1)
    
#     try:
#         success = init_gesture_recognition()
#         if success:
#             # Reset sentence
#             gesture_sentence = ""
            
#             # Start gesture recognition in background thread
#             gesture_thread = threading.Thread(target=update_gesture_recognition)
#             gesture_thread.daemon = True
#             gesture_thread.start()
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Gesture recognition started successfully'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to initialize camera. Please check if camera is available.'
#             })
#     except Exception as e:
#         print(f"❌ Error starting gesture recognition: {e}")
#         return jsonify({
#             'success': False,
#             'error': f'Error starting gesture recognition: {str(e)}'
#         })

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     """Stop the gesture recognition system"""
#     stop_gesture_recognition()
#     return jsonify({
#         'success': True,
#         'message': 'Gesture recognition stopped'
#     })

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route for gesture recognition"""
#     if not is_camera_running:
#         return "Camera not running", 400
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     """Get current gesture and sentence data"""
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'gesture_available': gesture_recognition_available,
#         'camera_running': is_camera_running
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     """Clear the recognized sentence"""
#     global gesture_sentence
#     gesture_sentence = ""
#     return jsonify({
#         'success': True,
#         'message': 'Sentence cleared'
#     })

# @app.route('/debug')
# def debug():
#     """Debug endpoint to check paths and available signs"""
#     debug_info = {
#         'base_dir': BASE_DIR,
#         'available_languages': list(SIGN_LANGUAGES.keys()),
#         'current_working_dir': os.getcwd(),
#         'gesture_recognition_available': gesture_recognition_available,
#         'gesture_recognition_initialized': gesture_recognizer is not None,
#         'camera_running': is_camera_running,
#         'current_gesture': current_gesture,
#         'gesture_sentence': gesture_sentence
#     }
    
#     for lang in SIGN_LANGUAGES:
#         signs_path = get_signs_path(lang)
#         debug_info[f'signs_path_{lang}'] = signs_path
#         debug_info[f'signs_exists_{lang}'] = os.path.exists(signs_path)
#         debug_info[f'available_signs_{lang}'] = get_words_in_database(lang)
    
#     return jsonify(debug_info)

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")
#     print(f"🎯 Gesture recognition available: {gesture_recognition_available}")
    
#     app.run(port=5001, debug=True)

# v4-trying old code(cam doesn't work but texttosign works) (is v1)
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# # Make sure punkt is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",           # American Sign Language
#     "bsl": "signs_bsl",       # British Sign Language  
#     "isl": "signs_isl"        # Indian Sign Language
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()

# # -------------------- Text to Sign Functions --------------------

# def get_signs_path(language):
#     """Get the path for the selected sign language."""
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     """List all available signs for the selected language."""
#     signs_path = get_signs_path(language)
#     print(f"🔍 Looking for signs in: {signs_path} for language: {language}")
    
#     if not os.path.exists(signs_path):
#         print(f"❌ Signs directory does not exist: {signs_path}")
#         return []
    
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     print(f"📹 Found {len(vid_names)} sign videos for {language}: {vid_names}")
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     """Find the closest match for a word or phrase in the signs database."""
#     # Try to find a direct match for the phrase
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
    
#     if phrase_match in available_signs:
#         return phrase_match
    
#     # If no direct match, fallback to individual word matching
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
    
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     signs_path = get_signs_path(language)
    
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 # Resize to consistent size
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#                 print(f"✅ Loaded: {sign} for {language}")
#             except Exception as e:
#                 print(f"❌ Error loading clip {sign_path}: {e}")
#                 return False
#         else:
#             print(f"⚠️ Missing file: {sign_path}")
#             return False

#     if clips:
#         try:
#             print(f"🎞️ Merging {len(clips)} clips for {language}...")
#             final_clip = concatenate_videoclips(clips, method="compose")
#             # Ensure output directory exists
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             # Write the video file
#             final_clip.write_videofile(
#                 output_path, 
#                 codec="libx264", 
#                 audio=False,
#                 fps=24
#             )
#             # Close clips to free memory
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             print(f"✅ Output saved to: {output_path}")
#             return True
#         except Exception as e:
#             print(f"❌ Error merging clips: {e}")
#             return False
#     else:
#         print("❌ No clips to merge.")
#         return False

# def text_to_sign(text, language):
#     """Main function to convert text to sign language video"""
#     print(f"🎯 Starting translation for: '{text}' in {language}")
    
#     # Sanitize the input
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     print(f"🧹 Sanitized Input: '{sanitized_text}'")

#     # Check for direct phrase matches first
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you", 
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
    
#     if sanitized_text in phrase_mappings:
#         phrase_sign = phrase_mappings[sanitized_text]
#         print(f"✅ Found phrase mapping: '{sanitized_text}' -> '{phrase_sign}'")
#         success = merge_signs([phrase_sign], language)
#         return success
    
#     # Word-by-word processing for other text
#     words = sanitized_text.split()
#     final_sequence = []

#     # Process each word
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             print(f"✅ '{w}' found in database as '{db_match}'")
#             final_sequence.append(db_match)
#         else:
#             print(f"❌ '{w}' not found in database, trying to spell it...")
#             spelled = spell_word(w, language)
#             if spelled:
#                 print(f"🔤 '{w}' → spelling as: {spelled}")
#                 final_sequence.extend(spelled)
#             else:
#                 print(f"❌ Cannot represent '{w}', skipping...")

#     print(f"📋 Final sequence for {language}: {final_sequence}")
    
#     if final_sequence:
#         success = merge_signs(final_sequence, language)
#         return success
#     else:
#         print("❌ No matching or spellable words found.")
#         return False

# # -------------------- Sign to Text (Gesture Recognition) Functions --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running
    
#     try:
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer()
#         camera = cv2.VideoCapture(0)
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         is_camera_running = True
#         print("✅ Gesture recognition initialized successfully")
#         return True
#     except Exception as e:
#         print(f"❌ Failed to initialize gesture recognition: {e}")
#         return False

# def update_gesture_recognition():
#     """Update gesture recognition in a separate thread"""
#     global current_gesture, gesture_sentence, last_gesture_time, is_camera_running
    
#     while is_camera_running:
#         if camera and gesture_recognizer:
#             ret, frame = camera.read()
#             if ret:
#                 frame = cv2.flip(frame, 1)
#                 processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
#                 current_gesture = detected_gesture
                
#                 # Add to sentence logic
#                 current_time = time.time()
#                 if (detected_gesture != current_gesture or 
#                     current_time - last_gesture_time > 2.0):
                    
#                     if detected_gesture not in ["No hand detected", "Unknown"]:
#                         gesture_sentence += detected_gesture + " "
#                         last_gesture_time = current_time

# def generate_frames():
#     """Generate video frames for streaming"""
#     global camera, gesture_recognizer, is_camera_running
    
#     while is_camera_running and camera:
#         ret, frame = camera.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)
        
#         # Process frame with gesture recognition
#         if gesture_recognizer:
#             processed_frame, _ = gesture_recognizer.recognize_gesture(frame)
            
#             # Add text overlay
#             height, width = processed_frame.shape[:2]
            
#             # Create overlay for text
#             overlay = processed_frame.copy()
#             cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#             cv2.rectangle(overlay, (10, height - 150), (width - 10, height - 10), (0, 0, 0), -1)
            
#             # Blend overlay with frame
#             processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
            
#             # Display current gesture
#             cv2.putText(processed_frame, f"Current Gesture: {current_gesture}", 
#                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
#             # Display sentence
#             cv2.putText(processed_frame, "Recognized Text:", 
#                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             # Word wrap for long sentences
#             sentence_words = gesture_sentence.split()
#             line_length = 40
#             lines = []
#             current_line = ""
            
#             for word in sentence_words:
#                 if len(current_line + word) <= line_length:
#                     current_line += word + " "
#                 else:
#                     lines.append(current_line.strip())
#                     current_line = word + " "
            
#             if current_line:
#                 lines.append(current_line.strip())
            
#             for i, line in enumerate(lines[-3:]):  # Show last 3 lines
#                 cv2.putText(processed_frame, line, 
#                            (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             # Display instructions
#             cv2.putText(processed_frame, "Hold gesture for 2 seconds to add to text", 
#                        (20, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
#             frame = processed_frame
        
#         # Encode frame as JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def stop_gesture_recognition():
#     """Stop gesture recognition and release resources"""
#     global is_camera_running, camera
    
#     is_camera_running = False
#     if camera:
#         camera.release()
#     cv2.destroyAllWindows()

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
            
#         print(f"📥 Received text to translate: '{text}' in {language}")
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'language': language,
#                 'message': f'Sign language video created for: {text} ({language.upper()})',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Could not generate {language.upper()} sign language video. Please try a different word.'
#             })
            
#     except Exception as e:
#         print(f"❌ Route error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def serve_video():
#     """Serve the generated text-to-sign video file"""
#     try:
#         if os.path.exists(STATIC_VIDEO_PATH):
#             return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Video not found'
#             }), 404
#     except Exception as e:
#         print(f"❌ Video serve error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# # -------------------- Gesture Recognition Routes --------------------

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     """Start the gesture recognition system"""
#     global gesture_sentence
    
#     try:
#         success = init_gesture_recognition()
#         if success:
#             # Start gesture recognition in background thread
#             thread = threading.Thread(target=update_gesture_recognition)
#             thread.daemon = True
#             thread.start()
            
#             gesture_sentence = ""  # Reset sentence
            
#             return jsonify({
#                 'success': True,
#                 'message': 'Gesture recognition started'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Failed to initialize camera or gesture recognition'
#             })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     """Stop the gesture recognition system"""
#     stop_gesture_recognition()
#     return jsonify({
#         'success': True,
#         'message': 'Gesture recognition stopped'
#     })

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route for gesture recognition"""
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     """Get current gesture and sentence data"""
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip()
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     """Clear the recognized sentence"""
#     global gesture_sentence
#     gesture_sentence = ""
#     return jsonify({
#         'success': True,
#         'message': 'Sentence cleared'
#     })

# @app.route('/debug')
# def debug():
#     """Debug endpoint to check paths and available signs"""
#     debug_info = {
#         'base_dir': BASE_DIR,
#         'available_languages': list(SIGN_LANGUAGES.keys()),
#         'current_working_dir': os.getcwd(),
#         'gesture_recognition_initialized': gesture_recognizer is not None,
#         'camera_running': is_camera_running
#     }
    
#     for lang in SIGN_LANGUAGES:
#         signs_path = get_signs_path(lang)
#         debug_info[f'signs_path_{lang}'] = signs_path
#         debug_info[f'signs_exists_{lang}'] = os.path.exists(signs_path)
#         debug_info[f'available_signs_{lang}'] = get_words_in_database(lang)
    
#     return jsonify(debug_info)

# @app.teardown_appcontext
# def close_camera(exception=None):
#     """Ensure camera is released when app closes"""
#     stop_gesture_recognition()

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")
    
#     for lang, folder in SIGN_LANGUAGES.items():
#         path = os.path.join(BASE_DIR, folder)
#         exists = os.path.exists(path)
#         print(f"📁 {lang.upper()} path: {path} - {'✅ Exists' if exists else '❌ Missing'}")
#         if exists:
#             signs = get_words_in_database(lang)
#             print(f"   📹 Available signs: {len(signs)}")
    
#     app.run(port=5001, debug=True)


# v5-trying to make signtotext work(camera opened but shut immediately i showed my hand)
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# # Make sure punkt is available
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl",
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# last_added_gesture = ""

# # -------------------- Text to Sign Functions (UNCHANGED) --------------------

# def get_signs_path(language):
#     """Get the path for the selected sign language."""
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     """List all available signs for the selected language."""
#     signs_path = get_signs_path(language)
    
#     if not os.path.exists(signs_path):
#         return []
    
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     """Find the closest match for a word or phrase in the signs database."""
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
    
#     if phrase_match in available_signs:
#         return phrase_match
    
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
    
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     signs_path = get_signs_path(language)
    
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 print(f"❌ Error loading clip {sign_path}: {e}")
#                 return False
#         else:
#             return False

#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             print(f"❌ Error merging clips: {e}")
#             return False
#     return False

# def text_to_sign(text, language):
#     """Main function to convert text to sign language video"""
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
    
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language)
    
#     words = sanitized_text.split()
#     final_sequence = []

#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
    
#     if final_sequence:
#         return merge_signs(final_sequence, language)
#     return False

# # -------------------- Sign to Text (FIXED Camera Access) --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running
    
#     print("🔧 Initializing gesture recognition system...")
    
#     try:
#         # Stop any existing camera
#         if camera is not None:
#             print("🔄 Releasing existing camera...")
#             try:
#                 camera.release()
#             except:
#                 pass
#             time.sleep(1.0)  # Give camera more time to release
#             camera = None
        
#         # Import and create gesture recognizer
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer()
        
#         # Initialize camera
#         print("📷 Opening camera...")
#         camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
#         if not camera.isOpened():
#             print("❌ Failed to open camera with DirectShow, trying default...")
#             camera = cv2.VideoCapture(0)
            
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        
#         # Test camera multiple times
#         for i in range(5):
#             ret, frame = camera.read()
#             if ret:
#                 print(f"✅ Camera test {i+1}/5 successful")
#                 break
#             print(f"⚠️ Camera test {i+1}/5 failed, retrying...")
#             time.sleep(0.2)
        
#         if not ret:
#             print("❌ Failed to read from camera after 5 attempts")
#             camera.release()
#             camera = None
#             return False
        
#         is_camera_running = True
#         print("✅ Gesture recognition initialized successfully")
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize gesture recognition: {e}")
#         import traceback
#         traceback.print_exc()
#         if camera:
#             camera.release()
#         camera = None
#         is_camera_running = False
#         return False

# def generate_frames():
#     """Generate video frames for streaming - SINGLE THREAD HANDLES EVERYTHING"""
#     global camera, gesture_recognizer, is_camera_running, current_gesture
#     global gesture_sentence, last_gesture_time, last_added_gesture
    
#     print("🎥 Starting frame generation...")
    
#     frame_count = 0
    
#     while is_camera_running:
#         try:
#             if not camera or not camera.isOpened():
#                 print("⚠️ Camera not available")
#                 time.sleep(0.5)
#                 continue
            
#             ret, frame = camera.read()
            
#             if not ret:
#                 print(f"⚠️ Failed to read frame {frame_count}")
#                 time.sleep(0.1)
#                 continue
            
#             frame_count += 1
#             frame = cv2.flip(frame, 1)
            
#             # Process frame with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Update current gesture
#                     old_gesture = current_gesture
#                     current_gesture = detected_gesture
                    
#                     # Add to sentence logic (every 2 seconds)
#                     current_time = time.time()
#                     time_since_last = current_time - last_gesture_time
                    
#                     if (detected_gesture != last_added_gesture and 
#                         time_since_last > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error"]):
                        
#                         gesture_sentence += detected_gesture + " "
#                         last_added_gesture = detected_gesture
#                         last_gesture_time = current_time
#                         print(f"📝 Added gesture: {detected_gesture} (sentence: {gesture_sentence.strip()})")
                    
#                     # Add overlays
#                     height, width = processed_frame.shape[:2]
                    
#                     # Semi-transparent overlays
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 80), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Current gesture text
#                     cv2.putText(processed_frame, f"Gesture: {current_gesture}", 
#                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
#                     # Recognized text
#                     cv2.putText(processed_frame, "Recognized:", 
#                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
#                     # Display sentence (word-wrapped)
#                     words = gesture_sentence.split()
#                     line_length = 35
#                     lines = []
#                     current_line = ""
                    
#                     for word in words:
#                         if len(current_line + word) <= line_length:
#                             current_line += word + " "
#                         else:
#                             lines.append(current_line.strip())
#                             current_line = word + " "
#                     if current_line:
#                         lines.append(current_line.strip())
                    
#                     for i, line in enumerate(lines[-2:]):
#                         cv2.putText(processed_frame, line, 
#                                    (20, 95 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
#                     # Instructions
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     frame = processed_frame
                    
#                 except Exception as e:
#                     print(f"❌ Error processing frame: {e}")
            
#             # Encode and yield frame
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
#         except Exception as e:
#             print(f"❌ Error in frame generation: {e}")
#             time.sleep(0.1)
    
#     print("🛑 Frame generation stopped")

# def stop_gesture_recognition():
#     """Stop gesture recognition and release resources"""
#     global is_camera_running, camera
    
#     print("🔄 Stopping gesture recognition...")
#     is_camera_running = False
#     time.sleep(0.5)  # Give threads time to exit
    
#     if camera:
#         try:
#             camera.release()
#         except:
#             pass
#         camera = None
    
#     cv2.destroyAllWindows()
#     print("✅ Gesture recognition stopped")

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'language': language,
#                 'message': f'Sign language video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': f'Could not generate video'
#             })
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     """Serve the generated video"""
#     try:
#         if os.path.exists(STATIC_VIDEO_PATH):
#             return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#         return jsonify({'success': False, 'error': 'Video not found'}), 404
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     """Start gesture recognition"""
#     global gesture_sentence, last_added_gesture
    
#     try:
#         success = init_gesture_recognition()
#         if success:
#             gesture_sentence = ""
#             last_added_gesture = ""
#             return jsonify({'success': True, 'message': 'Camera started'})
#         else:
#             return jsonify({'success': False, 'error': 'Failed to initialize camera'})
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     """Stop gesture recognition"""
#     stop_gesture_recognition()
#     return jsonify({'success': True, 'message': 'Camera stopped'})

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route"""
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     """Get current gesture and sentence"""
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'camera_running': is_camera_running
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     """Clear sentence"""
#     global gesture_sentence, last_added_gesture
#     gesture_sentence = ""
#     last_added_gesture = ""
#     return jsonify({'success': True, 'message': 'Cleared'})

# @app.route('/debug')
# def debug():
#     """Debug info"""
#     return jsonify({
#         'gesture_recognizer_loaded': gesture_recognizer is not None,
#         'camera_running': is_camera_running,
#         'camera_exists': camera is not None,
#         'current_gesture': current_gesture
#     })

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
    
#     for lang, folder in SIGN_LANGUAGES.items():
#         path = os.path.join(BASE_DIR, folder)
#         if os.path.exists(path):
#             signs = get_words_in_database(lang)
#             print(f"📁 {lang.upper()}: {len(signs)} signs available")
    
#     app.run(port=5001, debug=True, threaded=True)


# v6(trying to make signtotext work)(opens camera and places landmarks and can recognize)
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl",
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# last_added_gesture = ""
# output_frame = None
# frame_lock = threading.Lock()
# is_streaming = False  # NEW: Prevent multiple streams

# # -------------------- Text to Sign Functions (UNCHANGED) --------------------

# def get_signs_path(language):
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     signs_path = get_signs_path(language)
#     if not os.path.exists(signs_path):
#         return []
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
#     if phrase_match in available_signs:
#         return phrase_match
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     clips = []
#     signs_path = get_signs_path(language)
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 return False
#         else:
#             return False
#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             return False
#     return False

# def text_to_sign(text, language):
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language)
#     words = sanitized_text.split()
#     final_sequence = []
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
#     if final_sequence:
#         return merge_signs(final_sequence, language)
#     return False

# # -------------------- Sign to Text (FIXED - Single Stream Only) --------------------

# def init_gesture_recognition():
#     """Initialize gesture recognition system"""
#     global gesture_recognizer, camera, is_camera_running
    
#     print("🔧 Initializing gesture recognition system...")
    
#     try:
#         if camera is not None:
#             print("🔄 Releasing existing camera...")
#             try:
#                 camera.release()
#             except:
#                 pass
#             time.sleep(0.5)
#             camera = None
        
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer()
        
#         print("📷 Opening camera...")
#         camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if not camera.isOpened():
#             print("❌ Failed with DirectShow, trying default...")
#             camera = cv2.VideoCapture(0)
            
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Test camera
#         ret, frame = camera.read()
#         if not ret:
#             print("❌ Failed to read from camera")
#             camera.release()
#             camera = None
#             return False
        
#         is_camera_running = True
#         print("✅ Gesture recognition initialized")
        
#         # Start processing thread
#         thread = threading.Thread(target=process_camera_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize: {e}")
#         if camera:
#             camera.release()
#         camera = None
#         is_camera_running = False
#         return False

# def process_camera_frames():
#     """Process frames in background thread"""
#     global output_frame, current_gesture, gesture_sentence
#     global last_gesture_time, last_added_gesture
    
#     print("🎥 Starting frame processing...")
    
#     while is_camera_running:
#         try:
#             if not camera or not camera.isOpened():
#                 time.sleep(0.1)
#                 continue
            
#             ret, frame = camera.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             # Process with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Update gesture
#                     old_gesture = current_gesture
#                     current_gesture = detected_gesture
                    
#                     # # Add to sentence (every 2 seconds)
#                     # current_time = time.time()
#                     # if (detected_gesture != last_added_gesture and 
#                     #     current_time - last_gesture_time > 2.0 and
#                     #     detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error"]):
                        
#                     #     gesture_sentence += detected_gesture + " "
#                     #     last_added_gesture = detected_gesture
#                     #     last_gesture_time = current_time
#                     #     print(f"📝 Added: {detected_gesture}")
#                     # Add to sentence (every 2 seconds)
#                     current_time = time.time()
#                     if (current_time - last_gesture_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error"]):
#                         gesture_sentence += detected_gesture + " "
#                         last_added_gesture = detected_gesture
#                         last_gesture_time = current_time
#                         print(f"📝 Added: {detected_gesture}")
                    
#                     # Add overlays
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 100), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Text overlays
#                     cv2.putText(processed_frame, f"Gesture: {current_gesture}", 
#                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Text: {gesture_sentence.strip()}", 
#                                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     # Update output frame
#                     with frame_lock:
#                         output_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ Processing error: {e}")
#                     with frame_lock:
#                         output_frame = frame.copy()
#             else:
#                 with frame_lock:
#                     output_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 Frame processing stopped")

# def generate_frames():
#     """Generate frames for streaming - just yields processed frames"""
#     global is_streaming
    
#     if is_streaming:
#         print("⚠️ Already streaming, rejecting new connection")
#         return
    
#     is_streaming = True
#     print("📡 Starting video stream...")
    
#     try:
#         while is_camera_running:
#             with frame_lock:
#                 if output_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = output_frame.copy()
            
#             # Encode
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)  # ~30 FPS
#     finally:
#         is_streaming = False
#         print("📡 Stream ended")

# def stop_gesture_recognition():
#     """Stop gesture recognition"""
#     global is_camera_running, camera, output_frame
    
#     print("🔄 Stopping...")
#     is_camera_running = False
#     time.sleep(0.5)
    
#     if camera:
#         try:
#             camera.release()
#         except:
#             pass
#         camera = None
    
#     output_frame = None
#     cv2.destroyAllWindows()
#     print("✅ Stopped")

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'message': f'Video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Could not generate video'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     if os.path.exists(STATIC_VIDEO_PATH):
#         return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     global gesture_sentence, last_added_gesture
#     try:
#         success = init_gesture_recognition()
#         if success:
#             gesture_sentence = ""
#             last_added_gesture = ""
#             return jsonify({'success': True, 'message': 'Camera started'})
#         return jsonify({'success': False, 'error': 'Failed to start camera'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     stop_gesture_recognition()
#     return jsonify({'success': True})

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'camera_running': is_camera_running
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     global gesture_sentence, last_added_gesture
#     gesture_sentence = ""
#     last_added_gesture = ""
#     return jsonify({'success': True})

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     for lang in SIGN_LANGUAGES:
#         signs = get_words_in_database(lang)
#         print(f"📁 {lang.upper()}: {len(signs)} signs")
    
#     app.run(port=5001, debug=True, threaded=True)


# v7-trying to add options for signtotext
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl", 
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# last_added_gesture = ""
# output_frame = None
# frame_lock = threading.Lock()
# is_streaming = False
# current_gesture_language = "asl"  # Track current gesture language

# # -------------------- Text to Sign Functions (UNCHANGED) --------------------

# def get_signs_path(language):
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     signs_path = get_signs_path(language)
#     if not os.path.exists(signs_path):
#         return []
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
#     if phrase_match in available_signs:
#         return phrase_match
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     clips = []
#     signs_path = get_signs_path(language)
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 return False
#         else:
#             return False
#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             return False
#     return False

# def text_to_sign(text, language):
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language)
#     words = sanitized_text.split()
#     final_sequence = []
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
#     if final_sequence:
#         return merge_signs(final_sequence, language)
#     return False

# # -------------------- Sign to Text (FIXED - Proper language switching) --------------------

# def init_gesture_recognition(language="asl"):
#     """Initialize gesture recognition system with specified language"""
#     global gesture_recognizer, camera, is_camera_running, current_gesture_language
    
#     print(f"🔧 Initializing {language.upper()} gesture recognition system...")
#     current_gesture_language = language
    
#     try:
#         # Stop any existing camera first
#         if camera is not None:
#             print("🔄 Releasing existing camera...")
#             try:
#                 camera.release()
#             except:
#                 pass
#             time.sleep(1)  # Give time for camera to release
#             camera = None
        
#         # Clean up existing gesture recognizer
#         if gesture_recognizer is not None:
#             try:
#                 del gesture_recognizer
#             except:
#                 pass
#             gesture_recognizer = None
        
#         # Import and initialize the gesture recognizer
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer(language=language)
        
#         print("📷 Opening camera...")
#         camera = cv2.VideoCapture(0)
        
#         if not camera.isOpened():
#             print("❌ Failed to open camera with default backend, trying DirectShow...")
#             camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
#         if not camera.isOpened():
#             print("❌ Failed to open camera with any backend")
#             return False
            
#         # Set camera properties
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         camera.set(cv2.CAP_PROP_FPS, 30)
#         camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Test camera with multiple attempts
#         max_attempts = 5
#         for attempt in range(max_attempts):
#             ret, frame = camera.read()
#             if ret:
#                 print(f"✅ Camera test successful (attempt {attempt + 1})")
#                 break
#             else:
#                 print(f"⚠️ Camera test failed (attempt {attempt + 1}), retrying...")
#                 time.sleep(0.5)
#         else:
#             print("❌ Failed to read from camera after multiple attempts")
#             camera.release()
#             camera = None
#             return False
        
#         is_camera_running = True
#         print(f"✅ {language.upper()} gesture recognition initialized successfully")
        
#         # Reset gesture tracking
#         global gesture_sentence, last_added_gesture
#         gesture_sentence = ""
#         last_added_gesture = ""
        
#         # Start processing thread
#         thread = threading.Thread(target=process_camera_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize {language.upper()}: {e}")
#         import traceback
#         traceback.print_exc()
        
#         if camera:
#             try:
#                 camera.release()
#             except:
#                 pass
#             camera = None
#         is_camera_running = False
#         return False

# def process_camera_frames():
#     """Process frames in background thread"""
#     global output_frame, current_gesture, gesture_sentence, current_gesture_language
#     global last_gesture_time, last_added_gesture
    
#     print(f"🎥 Starting frame processing for {current_gesture_language.upper()}...")
    
#     while is_camera_running:
#         try:
#             if not camera or not camera.isOpened():
#                 print("⚠️ Camera not available, stopping processing...")
#                 break
            
#             ret, frame = camera.read()
#             if not ret:
#                 print("⚠️ Failed to read frame from camera")
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             # Process with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Update gesture
#                     current_gesture = detected_gesture
                    
#                     # Add to sentence (every 2 seconds)
#                     current_time = time.time()
#                     if (current_time - last_gesture_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"] and
#                         detected_gesture != last_added_gesture):
                        
#                         gesture_sentence += detected_gesture + " "
#                         last_added_gesture = detected_gesture
#                         last_gesture_time = current_time
#                         print(f"📝 Added {current_gesture_language.upper()} gesture: {detected_gesture}")
                    
#                     # Add overlays
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Text overlays with language info
#                     cv2.putText(processed_frame, f"Language: {current_gesture_language.upper()}", 
#                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Gesture: {current_gesture}", 
#                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Text: {gesture_sentence.strip()}", 
#                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     # Update output frame
#                     with frame_lock:
#                         output_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ Processing error: {e}")
#                     with frame_lock:
#                         output_frame = frame.copy()
#             else:
#                 with frame_lock:
#                     output_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 Frame processing stopped")

# def generate_frames():
#     """Generate frames for streaming - just yields processed frames"""
#     global is_streaming
    
#     if is_streaming:
#         print("⚠️ Already streaming, rejecting new connection")
#         return
    
#     is_streaming = True
#     print("📡 Starting video stream...")
    
#     try:
#         while is_camera_running:
#             with frame_lock:
#                 if output_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = output_frame.copy()
            
#             # Encode
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)  # ~30 FPS
#     except Exception as e:
#         print(f"❌ Stream error: {e}")
#     finally:
#         is_streaming = False
#         print("📡 Stream ended")

# def stop_gesture_recognition():
#     """Stop gesture recognition"""
#     global is_camera_running, camera, output_frame, gesture_recognizer
    
#     print("🔄 Stopping gesture recognition...")
#     is_camera_running = False
#     time.sleep(1)  # Give time for threads to stop
    
#     if camera:
#         try:
#             camera.release()
#             print("✅ Camera released")
#         except Exception as e:
#             print(f"❌ Error releasing camera: {e}")
#         camera = None
    
#     if gesture_recognizer:
#         try:
#             del gesture_recognizer
#             print("✅ Gesture recognizer cleaned up")
#         except Exception as e:
#             print(f"❌ Error cleaning up gesture recognizer: {e}")
#         gesture_recognizer = None
    
#     output_frame = None
#     cv2.destroyAllWindows()
#     print("✅ Gesture recognition stopped")

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'message': f'Video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Could not generate video'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     if os.path.exists(STATIC_VIDEO_PATH):
#         return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     global gesture_sentence, last_added_gesture
#     try:
#         language = request.args.get('language', 'asl')
#         if language not in ['asl', 'bsl', 'isl']:
#             language = 'asl'
            
#         print(f"🔄 Switching to {language.upper()} gesture recognition...")
        
#         # Stop any existing recognition first
#         stop_gesture_recognition()
#         time.sleep(2)  # Give more time for cleanup
        
#         success = init_gesture_recognition(language)
#         if success:
#             gesture_sentence = ""
#             last_added_gesture = ""
#             return jsonify({
#                 'success': True, 
#                 'message': f'{language.upper()} camera started', 
#                 'language': language.upper()
#             })
#         else:
#             return jsonify({'success': False, 'error': f'Failed to start {language.upper()} camera'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     stop_gesture_recognition()
#     return jsonify({'success': True, 'message': 'Camera stopped'})

# @app.route('/video_feed')
# def video_feed():
#     if not is_camera_running:
#         return jsonify({'error': 'Camera not running'}), 400
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'camera_running': is_camera_running,
#         'current_language': current_gesture_language.upper()
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     global gesture_sentence, last_added_gesture
#     gesture_sentence = ""
#     last_added_gesture = ""
#     return jsonify({'success': True, 'message': 'Text cleared'})

# @app.route('/debug')
# def debug_info():
#     return jsonify({
#         'text_to_sign_languages': list(SIGN_LANGUAGES.keys()),
#         'current_gesture_language': current_gesture_language,
#         'camera_status': 'running' if is_camera_running else 'stopped',
#         'gesture_sentence': gesture_sentence,
#         'is_streaming': is_streaming,
#         'gesture_recognizer_loaded': gesture_recognizer is not None,
#         'camera_loaded': camera is not None
#     })

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     for lang in SIGN_LANGUAGES:
#         signs = get_words_in_database(lang)
#         print(f"📁 {lang.upper()}: {len(signs)} signs")
    
#     print("\n🌐 Server starting on http://localhost:5001")
#     print("   Make sure to:")
#     print("   1. Allow camera access when prompted")
#     print("   2. Use 'Start Camera' button to begin gesture recognition")
#     print("   3. Switch languages using the dropdown")
    
#     app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)



# v8-trying to have options and print the same character twice in a row (works well!)
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl", 
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# last_added_gesture = ""
# output_frame = None
# frame_lock = threading.Lock()
# is_streaming = False
# current_gesture_language = "asl"  # Track current gesture language

# # -------------------- Text to Sign Functions (UNCHANGED) --------------------

# def get_signs_path(language):
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     signs_path = get_signs_path(language)
#     if not os.path.exists(signs_path):
#         return []
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
#     if phrase_match in available_signs:
#         return phrase_match
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     clips = []
#     signs_path = get_signs_path(language)
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 return False
#         else:
#             return False
#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             return False
#     return False

# def text_to_sign(text, language):
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language)
#     words = sanitized_text.split()
#     final_sequence = []
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
#     if final_sequence:
#         return merge_signs(final_sequence, language)
#     return False

# # -------------------- Sign to Text (Combined - Language options + gesture repetition) --------------------

# def init_gesture_recognition(language="asl"):
#     """Initialize gesture recognition system with specified language"""
#     global gesture_recognizer, camera, is_camera_running, current_gesture_language
    
#     print(f"🔧 Initializing {language.upper()} gesture recognition system...")
#     current_gesture_language = language
    
#     try:
#         # Stop any existing camera first
#         if camera is not None:
#             print("🔄 Releasing existing camera...")
#             try:
#                 camera.release()
#             except:
#                 pass
#             time.sleep(1)  # Give time for camera to release
#             camera = None
        
#         # Clean up existing gesture recognizer
#         if gesture_recognizer is not None:
#             try:
#                 del gesture_recognizer
#             except:
#                 pass
#             gesture_recognizer = None
        
#         # Import and initialize the gesture recognizer
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer(language=language)
        
#         print("📷 Opening camera...")
#         camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if not camera.isOpened():
#             print("❌ Failed with DirectShow, trying default...")
#             camera = cv2.VideoCapture(0)
            
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Test camera
#         ret, frame = camera.read()
#         if not ret:
#             print("❌ Failed to read from camera")
#             camera.release()
#             camera = None
#             return False
        
#         is_camera_running = True
#         print(f"✅ {language.upper()} gesture recognition initialized")
        
#         # Reset gesture tracking
#         global gesture_sentence, last_added_gesture
#         gesture_sentence = ""
#         last_added_gesture = ""
        
#         # Start processing thread
#         thread = threading.Thread(target=process_camera_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize {language.upper()}: {e}")
#         if camera:
#             camera.release()
#         camera = None
#         is_camera_running = False
#         return False

# def process_camera_frames():
#     """Process frames in background thread - with gesture repetition"""
#     global output_frame, current_gesture, gesture_sentence, current_gesture_language
#     global last_gesture_time, last_added_gesture
    
#     print(f"🎥 Starting frame processing for {current_gesture_language.upper()}...")
    
#     while is_camera_running:
#         try:
#             if not camera or not camera.isOpened():
#                 time.sleep(0.1)
#                 continue
            
#             ret, frame = camera.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             # Process with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Update gesture
#                     current_gesture = detected_gesture
                    
#                     # Add to sentence (every 2 seconds) - ALLOWS SAME GESTURE REPETITION
#                     current_time = time.time()
#                     if (current_time - last_gesture_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
#                         # Allow the same gesture to be added multiple times in a row
#                         gesture_sentence += detected_gesture + " "
#                         last_added_gesture = detected_gesture
#                         last_gesture_time = current_time
#                         print(f"📝 Added {current_gesture_language.upper()} gesture: {detected_gesture}")
                    
#                     # Add overlays
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Text overlays with language info
#                     cv2.putText(processed_frame, f"Language: {current_gesture_language.upper()}", 
#                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Gesture: {current_gesture}", 
#                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Text: {gesture_sentence.strip()}", 
#                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     # Update output frame
#                     with frame_lock:
#                         output_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ Processing error: {e}")
#                     with frame_lock:
#                         output_frame = frame.copy()
#             else:
#                 with frame_lock:
#                     output_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 Frame processing stopped")

# def generate_frames():
#     """Generate frames for streaming - just yields processed frames"""
#     global is_streaming
    
#     if is_streaming:
#         print("⚠️ Already streaming, rejecting new connection")
#         return
    
#     is_streaming = True
#     print("📡 Starting video stream...")
    
#     try:
#         while is_camera_running:
#             with frame_lock:
#                 if output_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = output_frame.copy()
            
#             # Encode
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)  # ~30 FPS
#     except Exception as e:
#         print(f"❌ Stream error: {e}")
#     finally:
#         is_streaming = False
#         print("📡 Stream ended")

# def stop_gesture_recognition():
#     """Stop gesture recognition"""
#     global is_camera_running, camera, output_frame, gesture_recognizer
    
#     print("🔄 Stopping gesture recognition...")
#     is_camera_running = False
#     time.sleep(1)  # Give time for threads to stop
    
#     if camera:
#         try:
#             camera.release()
#             print("✅ Camera released")
#         except Exception as e:
#             print(f"❌ Error releasing camera: {e}")
#         camera = None
    
#     if gesture_recognizer:
#         try:
#             del gesture_recognizer
#             print("✅ Gesture recognizer cleaned up")
#         except Exception as e:
#             print(f"❌ Error cleaning up gesture recognizer: {e}")
#         gesture_recognizer = None
    
#     output_frame = None
#     cv2.destroyAllWindows()
#     print("✅ Gesture recognition stopped")

# # -------------------- Flask Routes --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'message': f'Video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Could not generate video'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     if os.path.exists(STATIC_VIDEO_PATH):
#         return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     global gesture_sentence, last_added_gesture
#     try:
#         language = request.args.get('language', 'asl')
#         if language not in ['asl', 'bsl', 'isl']:
#             language = 'asl'
            
#         print(f"🔄 Switching to {language.upper()} gesture recognition...")
        
#         # Stop any existing recognition first
#         stop_gesture_recognition()
#         time.sleep(2)  # Give more time for cleanup
        
#         success = init_gesture_recognition(language)
#         if success:
#             gesture_sentence = ""
#             last_added_gesture = ""
#             return jsonify({
#                 'success': True, 
#                 'message': f'{language.upper()} camera started', 
#                 'language': language.upper()
#             })
#         else:
#             return jsonify({'success': False, 'error': f'Failed to start {language.upper()} camera'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     stop_gesture_recognition()
#     return jsonify({'success': True, 'message': 'Camera stopped'})

# @app.route('/video_feed')
# def video_feed():
#     if not is_camera_running:
#         return jsonify({'error': 'Camera not running'}), 400
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'camera_running': is_camera_running,
#         'current_language': current_gesture_language.upper()
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     global gesture_sentence, last_added_gesture
#     gesture_sentence = ""
#     last_added_gesture = ""
#     return jsonify({'success': True, 'message': 'Text cleared'})

# @app.route('/debug')
# def debug_info():
#     return jsonify({
#         'text_to_sign_languages': list(SIGN_LANGUAGES.keys()),
#         'current_gesture_language': current_gesture_language,
#         'camera_status': 'running' if is_camera_running else 'stopped',
#         'gesture_sentence': gesture_sentence,
#         'is_streaming': is_streaming,
#         'gesture_recognizer_loaded': gesture_recognizer is not None,
#         'camera_loaded': camera is not None
#     })

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     for lang in SIGN_LANGUAGES:
#         signs = get_words_in_database(lang)
#         print(f"📁 {lang.upper()}: {len(signs)} signs")
    
#     print("\n🌐 Server starting on http://localhost:5001")
#     print("   Features:")
#     print("   - Language options: ASL, BSL, ISL")
#     print("   - Same gesture can be converted to text multiple times in a row")
#     print("   - Real-time gesture recognition")
#     print("   - Text-to-sign video generation")
    
#     app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)



# v9-trying to make this bidirectional(works but has a few issues)(text to sign doesn't work)
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl", 
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# TRANSLATED_VIDEO_PATH = os.path.join("static", "translated_video.mp4")
# SIMILARITY_RATIO = 0.9

# # Global variables for gesture recognition
# gesture_recognizer = None
# camera = None
# is_camera_running = False
# current_gesture = "No gesture detected"
# gesture_sentence = ""
# last_gesture_time = time.time()
# last_added_gesture = ""
# output_frame = None
# frame_lock = threading.Lock()
# is_streaming = False
# current_gesture_language = "asl"  # Track current gesture language
# target_translation_language = "bsl"  # Default target language for translation

# # -------------------- Text to Sign Functions (MODIFIED for bidirectional) --------------------

# def get_signs_path(language):
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     signs_path = get_signs_path(language)
#     if not os.path.exists(signs_path):
#         return []
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
#     if phrase_match in available_signs:
#         return phrase_match
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
#     clips = []
#     signs_path = get_signs_path(language)
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 print(f"❌ Error loading sign {sign}: {e}")
#                 return False
#         else:
#             print(f"❌ Sign not found: {sign_path}")
#             return False
#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             print(f"❌ Error merging signs: {e}")
#             return False
#     return False

# def text_to_sign(text, language):
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language)
#     words = sanitized_text.split()
#     final_sequence = []
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
#     if final_sequence:
#         return merge_signs(final_sequence, language)
#     return False

# def translate_gesture_to_language(gesture_text, source_lang, target_lang):
#     """Translate a recognized gesture to another sign language"""
#     if not gesture_text or gesture_text in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing..."]:
#         return None
    
#     print(f"🔄 Translating '{gesture_text}' from {source_lang.upper()} to {target_lang.upper()}")
    
#     # Simple direct mapping - you can expand this with a proper translation dictionary
#     gesture_mapping = {
#         "asl": {
#             "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
#             "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
#             "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
#             "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
#             "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
#             "HELLO": "HELLO", "THANK YOU": "THANK YOU", "PLEASE": "PLEASE",
#             "YES": "YES", "NO": "NO", "SORRY": "SORRY", "HELP": "HELP"
#         },
#         "bsl": {
#             "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
#             "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
#             "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
#             "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
#             "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
#             "HELLO": "HELLO", "THANK YOU": "THANK YOU", "PLEASE": "PLEASE",
#             "YES": "YES", "NO": "NO", "SORRY": "SORRY", "HELP": "HELP"
#         },
#         "isl": {
#             "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
#             "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
#             "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
#             "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
#             "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
#             "HELLO": "NAMASTE", "THANK YOU": "DHANYAVAD", "PLEASE": "KRIPAYA",
#             "YES": "HAAN", "NO": "NAHI", "SORRY": "KSAMA", "HELP": "MADAD"
#         }
#     }
    
#     # Get the translated gesture
#     if (source_lang in gesture_mapping and 
#         target_lang in gesture_mapping and
#         gesture_text.upper() in gesture_mapping[source_lang]):
        
#         source_gesture = gesture_text.upper()
#         translated_gesture = gesture_mapping[target_lang].get(
#             gesture_mapping[source_lang][source_gesture], 
#             gesture_text.upper()
#         )
        
#         print(f"✅ Translated '{gesture_text}' to '{translated_gesture}' in {target_lang.upper()}")
#         return translated_gesture
    
#     return gesture_text.upper()

# # -------------------- Sign to Text (MODIFIED for bidirectional) --------------------

# def init_gesture_recognition(language="asl"):
#     """Initialize gesture recognition system with specified language"""
#     global gesture_recognizer, camera, is_camera_running, current_gesture_language
    
#     print(f"🔧 Initializing {language.upper()} gesture recognition system...")
#     current_gesture_language = language
    
#     try:
#         # Stop any existing camera first
#         if camera is not None:
#             print("🔄 Releasing existing camera...")
#             try:
#                 camera.release()
#             except:
#                 pass
#             time.sleep(1)
#             camera = None
        
#         # Clean up existing gesture recognizer
#         if gesture_recognizer is not None:
#             try:
#                 del gesture_recognizer
#             except:
#                 pass
#             gesture_recognizer = None
        
#         # Import and initialize the gesture recognizer
#         from gesture_recognition import GestureRecognizer
#         gesture_recognizer = GestureRecognizer(language=language)
        
#         print("📷 Opening camera...")
#         camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if not camera.isOpened():
#             print("❌ Failed with DirectShow, trying default...")
#             camera = cv2.VideoCapture(0)
            
#         if not camera.isOpened():
#             print("❌ Failed to open camera")
#             return False
            
#         camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Test camera
#         ret, frame = camera.read()
#         if not ret:
#             print("❌ Failed to read from camera")
#             camera.release()
#             camera = None
#             return False
        
#         is_camera_running = True
#         print(f"✅ {language.upper()} gesture recognition initialized")
        
#         # Reset gesture tracking
#         global gesture_sentence, last_added_gesture
#         gesture_sentence = ""
#         last_added_gesture = ""
        
#         # Start processing thread
#         thread = threading.Thread(target=process_camera_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize {language.upper()}: {e}")
#         if camera:
#             camera.release()
#         camera = None
#         is_camera_running = False
#         return False

# def process_camera_frames():
#     """Process frames in background thread - with bidirectional translation"""
#     global output_frame, current_gesture, gesture_sentence, current_gesture_language
#     global last_gesture_time, last_added_gesture, target_translation_language
    
#     print(f"🎥 Starting frame processing for {current_gesture_language.upper()}...")
    
#     while is_camera_running:
#         try:
#             if not camera or not camera.isOpened():
#                 time.sleep(0.1)
#                 continue
            
#             ret, frame = camera.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             # Process with gesture recognition
#             if gesture_recognizer:
#                 try:
#                     processed_frame, detected_gesture = gesture_recognizer.recognize_gesture(frame)
                    
#                     # Update gesture
#                     current_gesture = detected_gesture
                    
#                     # Add to sentence (every 2 seconds) - with bidirectional translation
#                     current_time = time.time()
#                     if (current_time - last_gesture_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
                        
#                         # Add the original gesture to sentence
#                         gesture_sentence += detected_gesture + " "
#                         last_added_gesture = detected_gesture
#                         last_gesture_time = current_time
                        
#                         # Generate translated video
#                         translated_gesture = translate_gesture_to_language(
#                             detected_gesture, 
#                             current_gesture_language, 
#                             target_translation_language
#                         )
                        
#                         if translated_gesture:
#                             # Create translated sign video
#                             success = merge_signs([translated_gesture], target_translation_language, TRANSLATED_VIDEO_PATH)
#                             if success:
#                                 print(f"🎬 Created {target_translation_language.upper()} video for: {translated_gesture}")
#                             else:
#                                 print(f"❌ Failed to create {target_translation_language.upper()} video for: {translated_gesture}")
                        
#                         print(f"📝 Added {current_gesture_language.upper()} gesture: {detected_gesture} -> {target_translation_language.upper()}")
                    
#                     # Add overlays with translation info
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     # Text overlays with language and translation info
#                     cv2.putText(processed_frame, f"Input: {current_gesture_language.upper()}", 
#                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Output: {target_translation_language.upper()}", 
#                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#                     cv2.putText(processed_frame, f"Gesture: {current_gesture}", 
#                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Text: {gesture_sentence.strip()}", 
#                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     # Update output frame
#                     with frame_lock:
#                         output_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ Processing error: {e}")
#                     with frame_lock:
#                         output_frame = frame.copy()
#             else:
#                 with frame_lock:
#                     output_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 Frame processing stopped")

# def generate_frames():
#     """Generate frames for streaming"""
#     global is_streaming
    
#     if is_streaming:
#         print("⚠️ Already streaming, rejecting new connection")
#         return
    
#     is_streaming = True
#     print("📡 Starting video stream...")
    
#     try:
#         while is_camera_running:
#             with frame_lock:
#                 if output_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = output_frame.copy()
            
#             # Encode
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)
#     except Exception as e:
#         print(f"❌ Stream error: {e}")
#     finally:
#         is_streaming = False
#         print("📡 Stream ended")

# def stop_gesture_recognition():
#     """Stop gesture recognition"""
#     global is_camera_running, camera, output_frame, gesture_recognizer
    
#     print("🔄 Stopping gesture recognition...")
#     is_camera_running = False
#     time.sleep(1)
    
#     if camera:
#         try:
#             camera.release()
#             print("✅ Camera released")
#         except Exception as e:
#             print(f"❌ Error releasing camera: {e}")
#         camera = None
    
#     if gesture_recognizer:
#         try:
#             del gesture_recognizer
#             print("✅ Gesture recognizer cleaned up")
#         except Exception as e:
#             print(f"❌ Error cleaning up gesture recognizer: {e}")
#         gesture_recognizer = None
    
#     output_frame = None
#     cv2.destroyAllWindows()
#     print("✅ Gesture recognition stopped")

# # -------------------- Flask Routes (MODIFIED for bidirectional) --------------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'message': f'Video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Could not generate video'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     if os.path.exists(STATIC_VIDEO_PATH):
#         return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# @app.route('/translated_video')
# def serve_translated_video():
#     if os.path.exists(TRANSLATED_VIDEO_PATH):
#         return send_file(TRANSLATED_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Translated video not found'}), 404

# @app.route('/start_gesture_recognition')
# def start_gesture_recognition():
#     global gesture_sentence, last_added_gesture
#     try:
#         language = request.args.get('language', 'asl')
#         if language not in ['asl', 'bsl', 'isl']:
#             language = 'asl'
            
#         print(f"🔄 Switching to {language.upper()} gesture recognition...")
        
#         # Stop any existing recognition first
#         stop_gesture_recognition()
#         time.sleep(2)
        
#         success = init_gesture_recognition(language)
#         if success:
#             gesture_sentence = ""
#             last_added_gesture = ""
#             return jsonify({
#                 'success': True, 
#                 'message': f'{language.upper()} camera started', 
#                 'language': language.upper()
#             })
#         else:
#             return jsonify({'success': False, 'error': f'Failed to start {language.upper()} camera'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/set_translation_language')
# def set_translation_language():
#     global target_translation_language
#     try:
#         language = request.args.get('language', 'bsl')
#         if language not in ['asl', 'bsl', 'isl']:
#             language = 'bsl'
            
#         target_translation_language = language
#         print(f"🎯 Translation target set to: {language.upper()}")
        
#         return jsonify({
#             'success': True, 
#             'message': f'Translation target set to {language.upper()}',
#             'language': language.upper()
#         })
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_gesture_recognition')
# def stop_gesture_recognition_route():
#     stop_gesture_recognition()
#     return jsonify({'success': True, 'message': 'Camera stopped'})

# @app.route('/video_feed')
# def video_feed():
#     if not is_camera_running:
#         return jsonify({'error': 'Camera not running'}), 400
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture_data')
# def get_gesture_data():
#     return jsonify({
#         'current_gesture': current_gesture,
#         'sentence': gesture_sentence.strip(),
#         'camera_running': is_camera_running,
#         'current_language': current_gesture_language.upper(),
#         'target_language': target_translation_language.upper()
#     })

# @app.route('/clear_gesture_sentence')
# def clear_gesture_sentence():
#     global gesture_sentence, last_added_gesture
#     gesture_sentence = ""
#     last_added_gesture = ""
#     return jsonify({'success': True, 'message': 'Text cleared'})

# @app.route('/debug')
# def debug_info():
#     return jsonify({
#         'text_to_sign_languages': list(SIGN_LANGUAGES.keys()),
#         'current_gesture_language': current_gesture_language,
#         'target_translation_language': target_translation_language,
#         'camera_status': 'running' if is_camera_running else 'stopped',
#         'gesture_sentence': gesture_sentence,
#         'is_streaming': is_streaming,
#         'gesture_recognizer_loaded': gesture_recognizer is not None,
#         'camera_loaded': camera is not None
#     })

# if __name__ == '__main__':
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
#     os.makedirs(os.path.dirname(TRANSLATED_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     for lang in SIGN_LANGUAGES:
#         signs = get_words_in_database(lang)
#         print(f"📁 {lang.upper()}: {len(signs)} signs")
    
#     print("\n🌐 Server starting on http://localhost:5001")
#     print("   NEW: Bidirectional Translation Feature")
#     print("   - Gesture in one language, get translation in another")
#     print("   - Real-time cross-language sign generation")
#     print("   - Multiple language pair support")
    
#     app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)


# v10 -to have both bidirectional and have individual sign to text, works well!
# from flask import Flask, render_template, request, jsonify, send_file, Response
# import sys
# import os
# import nltk
# import re
# import cv2
# import time
# import threading
# from nltk.tokenize import word_tokenize
# from difflib import SequenceMatcher
# from moviepy import VideoFileClip, concatenate_videoclips

# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt")

# # CONSTANTS
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SIGN_LANGUAGES = {
#     "asl": "signs",
#     "bsl": "signs_bsl", 
#     "isl": "signs_isl"
# }
# DEFAULT_LANGUAGE = "asl"
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# # Global variables for INDIVIDUAL sign to text
# individual_recognizer = None
# individual_camera = None
# is_individual_running = False
# individual_gesture = "No gesture detected"
# individual_sentence = ""
# last_individual_time = time.time()
# last_individual_gesture = ""
# individual_frame = None
# individual_frame_lock = threading.Lock()
# is_individual_streaming = False
# individual_language = "asl"

# # Global variables for BIDIRECTIONAL translation
# bidirectional_recognizer = None
# bidirectional_camera = None
# is_bidirectional_running = False
# bidirectional_gesture = "No gesture detected"
# bidirectional_text = ""
# last_bidirectional_time = time.time()
# bidirectional_frame = None
# bidirectional_frame_lock = threading.Lock()
# is_bidirectional_streaming = False
# source_language = "asl"
# target_language = "bsl"
# translated_video_queue = []  # Queue to store generated videos
# current_translated_video = None
# video_queue_lock = threading.Lock()

# # ==================== TEXT TO SIGN ====================

# def get_signs_path(language):
#     folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
#     return os.path.join(BASE_DIR, folder_name)

# def get_words_in_database(language):
#     signs_path = get_signs_path(language)
#     if not os.path.exists(signs_path):
#         return []
#     vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w, language):
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database(language)
#     if phrase_match in available_signs:
#         return phrase_match
#     best_score = -1.0
#     best_vid_name = None
#     for v in available_signs:
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word, language):
#     available_signs = get_words_in_database(language)
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             return None
#     return spelled

# def merge_signs(sign_sequence, language, output_path):
#     clips = []
#     signs_path = get_signs_path(language)
#     for sign in sign_sequence:
#         sign_path = os.path.join(signs_path, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#             except Exception as e:
#                 print(f"❌ Error loading sign {sign}: {e}")
#                 return False
#         else:
#             print(f"❌ Sign not found: {sign_path}")
#             return False
#     if clips:
#         try:
#             final_clip = concatenate_videoclips(clips, method="compose")
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             final_clip.write_videofile(output_path, codec="libx264", audio=False, fps=24)
#             for clip in clips:
#                 clip.close()
#             final_clip.close()
#             return True
#         except Exception as e:
#             print(f"❌ Error merging signs: {e}")
#             return False
#     return False

# def text_to_sign(text, language, output_path):
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
#     phrase_mappings = {
#         "what is your name": "what_is_your_name",
#         "how are you": "how_are_you",
#         "im great": "im_great",
#         "my name is": "my_name_is",
#         "good morning": "goodmorning"
#     }
#     if sanitized_text in phrase_mappings:
#         return merge_signs([phrase_mappings[sanitized_text]], language, output_path)
#     words = sanitized_text.split()
#     final_sequence = []
#     for w in words:
#         db_match = find_in_db(w, language)
#         if db_match:
#             final_sequence.append(db_match)
#         else:
#             spelled = spell_word(w, language)
#             if spelled:
#                 final_sequence.extend(spelled)
#     if final_sequence:
#         return merge_signs(final_sequence, language, output_path)
#     return False

# # ==================== INDIVIDUAL SIGN TO TEXT ====================

# def init_individual_recognition(language="asl"):
#     """Initialize INDIVIDUAL sign to text recognition"""
#     global individual_recognizer, individual_camera, is_individual_running, individual_language
    
#     print(f"🔧 [INDIVIDUAL] Initializing {language.upper()} sign recognition...")
#     individual_language = language
    
#     try:
#         if individual_camera is not None:
#             print("🔄 [INDIVIDUAL] Releasing existing camera...")
#             try:
#                 individual_camera.release()
#             except:
#                 pass
#             time.sleep(1)
#             individual_camera = None
        
#         if individual_recognizer is not None:
#             try:
#                 del individual_recognizer
#             except:
#                 pass
#             individual_recognizer = None
        
#         from gesture_recognition import GestureRecognizer
#         individual_recognizer = GestureRecognizer(language=language)
        
#         print("📷 [INDIVIDUAL] Opening camera...")
#         individual_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if not individual_camera.isOpened():
#             print("❌ [INDIVIDUAL] Failed with DirectShow, trying default...")
#             individual_camera = cv2.VideoCapture(0)
            
#         if not individual_camera.isOpened():
#             print("❌ [INDIVIDUAL] Failed to open camera")
#             return False
            
#         individual_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         individual_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         individual_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         ret, frame = individual_camera.read()
#         if not ret:
#             print("❌ [INDIVIDUAL] Failed to read from camera")
#             individual_camera.release()
#             individual_camera = None
#             return False
        
#         is_individual_running = True
#         print(f"✅ [INDIVIDUAL] {language.upper()} recognition initialized")
        
#         global individual_sentence, last_individual_gesture
#         individual_sentence = ""
#         last_individual_gesture = ""
        
#         thread = threading.Thread(target=process_individual_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ [INDIVIDUAL] Failed to initialize: {e}")
#         if individual_camera:
#             individual_camera.release()
#         individual_camera = None
#         is_individual_running = False
#         return False

# def process_individual_frames():
#     """Process frames for INDIVIDUAL sign to text"""
#     global individual_frame, individual_gesture, individual_sentence, individual_language
#     global last_individual_time, last_individual_gesture
    
#     print(f"🎥 [INDIVIDUAL] Starting frame processing...")
    
#     while is_individual_running:
#         try:
#             if not individual_camera or not individual_camera.isOpened():
#                 time.sleep(0.1)
#                 continue
            
#             ret, frame = individual_camera.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             if individual_recognizer:
#                 try:
#                     processed_frame, detected_gesture = individual_recognizer.recognize_gesture(frame)
#                     individual_gesture = detected_gesture
                    
#                     current_time = time.time()
#                     if (current_time - last_individual_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
#                         individual_sentence += detected_gesture + " "
#                         last_individual_gesture = detected_gesture
#                         last_individual_time = current_time
#                         print(f"📝 [INDIVIDUAL] Added: {detected_gesture}")
                    
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     cv2.putText(processed_frame, f"Mode: Individual Sign to Text", 
#                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Language: {individual_language.upper()}", 
#                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#                     cv2.putText(processed_frame, f"Gesture: {individual_gesture}", 
#                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Text: {individual_sentence.strip()}", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
#                     with individual_frame_lock:
#                         individual_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ [INDIVIDUAL] Processing error: {e}")
#                     with individual_frame_lock:
#                         individual_frame = frame.copy()
#             else:
#                 with individual_frame_lock:
#                     individual_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ [INDIVIDUAL] Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 [INDIVIDUAL] Frame processing stopped")

# def generate_individual_frames():
#     """Generate frames for INDIVIDUAL streaming"""
#     global is_individual_streaming
    
#     if is_individual_streaming:
#         print("⚠️ [INDIVIDUAL] Already streaming")
#         return
    
#     is_individual_streaming = True
#     print("📡 [INDIVIDUAL] Starting stream...")
    
#     try:
#         while is_individual_running:
#             with individual_frame_lock:
#                 if individual_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = individual_frame.copy()
            
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)
#     except Exception as e:
#         print(f"❌ [INDIVIDUAL] Stream error: {e}")
#     finally:
#         is_individual_streaming = False
#         print("📡 [INDIVIDUAL] Stream ended")

# def stop_individual_recognition():
#     """Stop INDIVIDUAL recognition"""
#     global is_individual_running, individual_camera, individual_frame, individual_recognizer
    
#     print("🔄 [INDIVIDUAL] Stopping recognition...")
#     is_individual_running = False
#     time.sleep(1)
    
#     if individual_camera:
#         try:
#             individual_camera.release()
#             print("✅ [INDIVIDUAL] Camera released")
#         except Exception as e:
#             print(f"❌ [INDIVIDUAL] Error releasing camera: {e}")
#         individual_camera = None
    
#     if individual_recognizer:
#         try:
#             del individual_recognizer
#             print("✅ [INDIVIDUAL] Recognizer cleaned up")
#         except Exception as e:
#             print(f"❌ [INDIVIDUAL] Error cleaning up: {e}")
#         individual_recognizer = None
    
#     individual_frame = None
#     cv2.destroyAllWindows()
#     print("✅ [INDIVIDUAL] Stopped")

# # ==================== BIDIRECTIONAL TRANSLATION ====================

# def translate_gesture_to_language(gesture_text, source_lang, target_lang):
#     """Translate gesture between sign languages"""
#     if not gesture_text or gesture_text in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]:
#         return None
    
#     gesture_text_upper = gesture_text.upper()
    
#     # For alphabets and common signs
#     if len(gesture_text_upper) == 1 and gesture_text_upper.isalpha():
#         return gesture_text_upper.lower()
    
#     # ISL specific mappings
#     if target_lang == "isl":
#         isl_mappings = {
#             "hello": "namaste",
#             "thank you": "dhanyavad",
#             "please": "kripaya",
#             "yes": "haan",
#             "no": "nahi",
#             "sorry": "ksama",
#             "help": "madad"
#         }
#         if gesture_text.lower() in isl_mappings:
#             return isl_mappings[gesture_text.lower()]
    
#     return gesture_text.lower()

# def init_bidirectional_recognition(source_lang="asl", target_lang="bsl"):
#     """Initialize BIDIRECTIONAL translation"""
#     global bidirectional_recognizer, bidirectional_camera, is_bidirectional_running
#     global source_language, target_language
    
#     print(f"🔧 [BIDIRECTIONAL] Initializing: {source_lang.upper()} → {target_lang.upper()}...")
#     source_language = source_lang
#     target_language = target_lang
    
#     try:
#         if bidirectional_camera is not None:
#             print("🔄 [BIDIRECTIONAL] Releasing existing camera...")
#             try:
#                 bidirectional_camera.release()
#             except:
#                 pass
#             time.sleep(1)
#             bidirectional_camera = None
        
#         if bidirectional_recognizer is not None:
#             try:
#                 del bidirectional_recognizer
#             except:
#                 pass
#             bidirectional_recognizer = None
        
#         from gesture_recognition import GestureRecognizer
#         bidirectional_recognizer = GestureRecognizer(language=source_lang)
        
#         print("📷 [BIDIRECTIONAL] Opening camera...")
#         bidirectional_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
#         if not bidirectional_camera.isOpened():
#             print("❌ [BIDIRECTIONAL] Failed with DirectShow, trying default...")
#             bidirectional_camera = cv2.VideoCapture(0)
            
#         if not bidirectional_camera.isOpened():
#             print("❌ [BIDIRECTIONAL] Failed to open camera")
#             return False
            
#         bidirectional_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         bidirectional_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         bidirectional_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         ret, frame = bidirectional_camera.read()
#         if not ret:
#             print("❌ [BIDIRECTIONAL] Failed to read from camera")
#             bidirectional_camera.release()
#             bidirectional_camera = None
#             return False
        
#         is_bidirectional_running = True
#         print(f"✅ [BIDIRECTIONAL] Translation initialized")
        
#         global bidirectional_text, translated_video_queue, current_translated_video
#         bidirectional_text = ""
#         translated_video_queue = []
#         current_translated_video = None
        
#         thread = threading.Thread(target=process_bidirectional_frames, daemon=True)
#         thread.start()
        
#         return True
        
#     except Exception as e:
#         print(f"❌ [BIDIRECTIONAL] Failed to initialize: {e}")
#         if bidirectional_camera:
#             bidirectional_camera.release()
#         bidirectional_camera = None
#         is_bidirectional_running = False
#         return False

# def process_bidirectional_frames():
#     """Process frames for BIDIRECTIONAL translation"""
#     global bidirectional_frame, bidirectional_gesture, bidirectional_text
#     global last_bidirectional_time, translated_video_queue, current_translated_video
    
#     print(f"🎥 [BIDIRECTIONAL] Starting processing: {source_language.upper()} → {target_language.upper()}...")
    
#     while is_bidirectional_running:
#         try:
#             if not bidirectional_camera or not bidirectional_camera.isOpened():
#                 time.sleep(0.1)
#                 continue
            
#             ret, frame = bidirectional_camera.read()
#             if not ret:
#                 time.sleep(0.1)
#                 continue
            
#             frame = cv2.flip(frame, 1)
            
#             if bidirectional_recognizer:
#                 try:
#                     processed_frame, detected_gesture = bidirectional_recognizer.recognize_gesture(frame)
#                     bidirectional_gesture = detected_gesture
                    
#                     current_time = time.time()
#                     if (current_time - last_bidirectional_time > 2.0 and
#                         detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
                        
#                         bidirectional_text += detected_gesture + " "
#                         last_bidirectional_time = current_time
                        
#                         # Generate translated video with unique filename
#                         translated_gesture = translate_gesture_to_language(
#                             detected_gesture, 
#                             source_language, 
#                             target_language
#                         )
                        
#                         if translated_gesture:
#                             timestamp = int(time.time() * 1000)
#                             video_filename = f"translated_{timestamp}.mp4"
#                             video_path = os.path.join("static", video_filename)
                            
#                             success = merge_signs([translated_gesture], target_language, video_path)
#                             if success:
#                                 with video_queue_lock:
#                                     translated_video_queue.append({
#                                         'gesture': detected_gesture,
#                                         'translated': translated_gesture,
#                                         'filename': video_filename,
#                                         'path': video_path,
#                                         'timestamp': timestamp
#                                     })
#                                     # Keep only last 10 videos
#                                     if len(translated_video_queue) > 10:
#                                         old_video = translated_video_queue.pop(0)
#                                         try:
#                                             if os.path.exists(old_video['path']):
#                                                 os.remove(old_video['path'])
#                                         except:
#                                             pass
#                                     current_translated_video = translated_video_queue[-1]
                                
#                                 print(f"🎬 [BIDIRECTIONAL] Created video: {detected_gesture} → {translated_gesture} ({target_language.upper()})")
#                             else:
#                                 print(f"❌ [BIDIRECTIONAL] Failed to create video for: {translated_gesture}")
                    
#                     height, width = processed_frame.shape[:2]
#                     overlay = processed_frame.copy()
#                     cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
#                     cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
#                     processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
#                     cv2.putText(processed_frame, f"Mode: Bidirectional Translation", 
#                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
#                     cv2.putText(processed_frame, f"Input: {source_language.upper()}", 
#                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#                     cv2.putText(processed_frame, f"Output: {target_language.upper()}", 
#                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#                     cv2.putText(processed_frame, f"Gesture: {bidirectional_gesture}", 
#                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
#                                (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
#                     with bidirectional_frame_lock:
#                         bidirectional_frame = processed_frame.copy()
                        
#                 except Exception as e:
#                     print(f"❌ [BIDIRECTIONAL] Processing error: {e}")
#                     with bidirectional_frame_lock:
#                         bidirectional_frame = frame.copy()
#             else:
#                 with bidirectional_frame_lock:
#                     bidirectional_frame = frame.copy()
            
#         except Exception as e:
#             print(f"❌ [BIDIRECTIONAL] Frame error: {e}")
#             time.sleep(0.1)
    
#     print("🛑 [BIDIRECTIONAL] Processing stopped")

# def generate_bidirectional_frames():
#     """Generate frames for BIDIRECTIONAL streaming"""
#     global is_bidirectional_streaming
    
#     if is_bidirectional_streaming:
#         print("⚠️ [BIDIRECTIONAL] Already streaming")
#         return
    
#     is_bidirectional_streaming = True
#     print("📡 [BIDIRECTIONAL] Starting stream...")
    
#     try:
#         while is_bidirectional_running:
#             with bidirectional_frame_lock:
#                 if bidirectional_frame is None:
#                     time.sleep(0.1)
#                     continue
#                 frame = bidirectional_frame.copy()
            
#             ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#             if not ret:
#                 continue
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
#             time.sleep(0.033)
#     except Exception as e:
#         print(f"❌ [BIDIRECTIONAL] Stream error: {e}")
#     finally:
#         is_bidirectional_streaming = False
#         print("📡 [BIDIRECTIONAL] Stream ended")

# def stop_bidirectional_recognition():
#     """Stop BIDIRECTIONAL translation"""
#     global is_bidirectional_running, bidirectional_camera, bidirectional_frame, bidirectional_recognizer
    
#     print("🔄 [BIDIRECTIONAL] Stopping...")
#     is_bidirectional_running = False
#     time.sleep(1)
    
#     if bidirectional_camera:
#         try:
#             bidirectional_camera.release()
#             print("✅ [BIDIRECTIONAL] Camera released")
#         except Exception as e:
#             print(f"❌ [BIDIRECTIONAL] Error releasing camera: {e}")
#         bidirectional_camera = None
    
#     if bidirectional_recognizer:
#         try:
#             del bidirectional_recognizer
#             print("✅ [BIDIRECTIONAL] Recognizer cleaned up")
#         except Exception as e:
#             print(f"❌ [BIDIRECTIONAL] Error cleaning up: {e}")
#         bidirectional_recognizer = None
    
#     bidirectional_frame = None
#     cv2.destroyAllWindows()
#     print("✅ [BIDIRECTIONAL] Stopped")

# # ==================== FLASK ROUTES ====================

# @app.route('/')
# def index():
#     return render_template('index.html')

# # ===== TEXT TO SIGN ROUTES =====
# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
#         language = data.get('language', 'asl')
        
#         if not text:
#             return jsonify({'success': False, 'error': 'No text provided'})
        
#         if language not in SIGN_LANGUAGES:
#             language = DEFAULT_LANGUAGE
        
#         success = text_to_sign(text, language, STATIC_VIDEO_PATH)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'message': f'Video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Could not generate video'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/video')
# def serve_video():
#     if os.path.exists(STATIC_VIDEO_PATH):
#         return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# # ===== INDIVIDUAL SIGN TO TEXT ROUTES =====
# @app.route('/start_individual')
# def start_individual():
#     global individual_sentence, last_individual_gesture
#     try:
#         language = request.args.get('language', 'asl')
#         if language not in ['asl', 'bsl', 'isl']:
#             language = 'asl'
            
#         print(f"🔄 [INDIVIDUAL] Starting {language.upper()}...")
        
#         stop_individual_recognition()
#         time.sleep(2)
        
#         success = init_individual_recognition(language)
#         if success:
#             individual_sentence = ""
#             last_individual_gesture = ""
#             return jsonify({
#                 'success': True, 
#                 'message': f'{language.upper()} Individual recognition started', 
#                 'language': language.upper()
#             })
#         else:
#             return jsonify({'success': False, 'error': f'Failed to start {language.upper()}'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_individual')
# def stop_individual():
#     stop_individual_recognition()
#     return jsonify({'success': True, 'message': 'Individual recognition stopped'})

# @app.route('/individual_feed')
# def individual_feed():
#     if not is_individual_running:
#         return jsonify({'error': 'Individual recognition not running'}), 400
#     return Response(generate_individual_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_individual_data')
# def get_individual_data():
#     return jsonify({
#         'current_gesture': individual_gesture,
#         'sentence': individual_sentence.strip(),
#         'running': is_individual_running,
#         'language': individual_language.upper()
#     })

# @app.route('/clear_individual')
# def clear_individual():
#     global individual_sentence, last_individual_gesture
#     individual_sentence = ""
#     last_individual_gesture = ""
#     return jsonify({'success': True, 'message': 'Text cleared'})

# # ===== BIDIRECTIONAL TRANSLATION ROUTES =====
# @app.route('/start_bidirectional')
# def start_bidirectional():
#     global bidirectional_text, translated_video_queue, current_translated_video
#     try:
#         source = request.args.get('source', 'asl')
#         target = request.args.get('target', 'bsl')
        
#         if source not in ['asl', 'bsl', 'isl']:
#             source = 'asl'
#         if target not in ['asl', 'bsl', 'isl']:
#             target = 'bsl'
            
#         print(f"🔄 [BIDIRECTIONAL] Starting: {source.upper()} → {target.upper()}...")
        
#         stop_bidirectional_recognition()
#         time.sleep(2)
        
#         success = init_bidirectional_recognition(source, target)
#         if success:
#             bidirectional_text = ""
#             translated_video_queue = []
#             current_translated_video = None
#             return jsonify({
#                 'success': True, 
#                 'message': f'Translation started: {source.upper()} → {target.upper()}',
#                 'source': source.upper(),
#                 'target': target.upper()
#             })
#         else:
#             return jsonify({'success': False, 'error': 'Failed to start translation'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/stop_bidirectional')
# def stop_bidirectional():
#     stop_bidirectional_recognition()
#     return jsonify({'success': True, 'message': 'Translation stopped'})

# @app.route('/bidirectional_feed')
# def bidirectional_feed():
#     if not is_bidirectional_running:
#         return jsonify({'error': 'Translation not running'}), 400
#     return Response(generate_bidirectional_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_bidirectional_data')
# def get_bidirectional_data():
#     with video_queue_lock:
#         latest_video = current_translated_video.copy() if current_translated_video else None
    
#     return jsonify({
#         'current_gesture': bidirectional_gesture,
#         'text': bidirectional_text.strip(),
#         'running': is_bidirectional_running,
#         'source_language': source_language.upper(),
#         'target_language': target_language.upper(),
#         'latest_video': latest_video,
#         'video_count': len(translated_video_queue)
#     })

# @app.route('/clear_bidirectional')
# def clear_bidirectional():
#     global bidirectional_text
#     bidirectional_text = ""
#     return jsonify({'success': True, 'message': 'Translation text cleared'})

# @app.route('/get_translated_videos')
# def get_translated_videos():
#     """Get list of all translated videos"""
#     with video_queue_lock:
#         videos = translated_video_queue.copy()
#     return jsonify({
#         'success': True,
#         'videos': videos
#     })

# @app.route('/translated_video/<filename>')
# def serve_translated_video(filename):
#     """Serve a specific translated video"""
#     video_path = os.path.join("static", filename)
#     if os.path.exists(video_path):
#         return send_file(video_path, as_attachment=False, mimetype='video/mp4')
#     return jsonify({'error': 'Video not found'}), 404

# # ===== DEBUG ROUTE =====
# @app.route('/debug')
# def debug_info():
#     return jsonify({
#         'text_to_sign': {
#             'languages': list(SIGN_LANGUAGES.keys()),
#             'output_path': STATIC_VIDEO_PATH
#         },
#         'individual': {
#             'language': individual_language,
#             'status': 'running' if is_individual_running else 'stopped',
#             'sentence': individual_sentence,
#             'streaming': is_individual_streaming
#         },
#         'bidirectional': {
#             'source': source_language,
#             'target': target_language,
#             'status': 'running' if is_bidirectional_running else 'stopped',
#             'text': bidirectional_text,
#             'streaming': is_bidirectional_streaming,
#             'video_count': len(translated_video_queue)
#         }
#     })

# if __name__ == '__main__':
#     os.makedirs("static", exist_ok=True)
    
#     print("🚀 Starting Complete Sign Language Translation App...")
#     print("\n📁 Available Sign Languages:")
#     for lang in SIGN_LANGUAGES:
#         signs = get_words_in_database(lang)
#         print(f"   - {lang.upper()}: {len(signs)} signs")
    
#     print("\n🌐 Server starting on http://localhost:5001")
#     print("\n✨ Features:")
#     print("   1️⃣  Text to Sign - Convert text to sign language videos")
#     print("   2️⃣  Individual Sign to Text - Simple sign recognition → text")
#     print("   3️⃣  Bidirectional Translation - Sign in one language → video in another")
#     print("\n   Supported Languages: ASL, BSL, ISL")
#     print("\n" + "="*60)
    
#     app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)


# v11-to fix the output issue for bidirectional and making sure both hands are detected for all languages
from flask import Flask, render_template, request, jsonify, send_file, Response
import sys
import os
import nltk
import re
import cv2
import time
import threading
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from moviepy import VideoFileClip, concatenate_videoclips

sys.path.append(os.path.dirname(__file__))

app = Flask(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGN_LANGUAGES = {
    "asl": "signs",
    "bsl": "signs_bsl", 
    "isl": "signs_isl"
}
DEFAULT_LANGUAGE = "asl"
SIMILARITY_RATIO = 0.9

# Global variables for INDIVIDUAL sign to text
individual_recognizer = None
individual_camera = None
is_individual_running = False
individual_gesture = "No gesture detected"
individual_sentence = ""
last_individual_time = time.time()
last_individual_gesture = ""
individual_frame = None
individual_frame_lock = threading.Lock()
is_individual_streaming = False
individual_language = "asl"

# Global variables for BIDIRECTIONAL translation
bidirectional_recognizer = None
bidirectional_camera = None
is_bidirectional_running = False
bidirectional_gesture = "No gesture detected"
bidirectional_text = ""
last_bidirectional_time = time.time()
bidirectional_frame = None
bidirectional_frame_lock = threading.Lock()
is_bidirectional_streaming = False
source_language = "asl"
target_language = "bsl"
translated_video_queue = []  # Queue to store generated videos
current_translated_video = None
video_queue_lock = threading.Lock()
gesture_buffer = []  # Buffer to store gestures for combined video
COMBINE_AFTER_GESTURES = 7  # Combine videos after 7 gestures

# ==================== TEXT TO SIGN ====================

def get_signs_path(language):
    folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
    return os.path.join(BASE_DIR, folder_name)

def get_words_in_database(language):
    signs_path = get_signs_path(language)
    if not os.path.exists(signs_path):
        return []
    vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
    vid_names = [v[:-4].lower() for v in vids]
    return vid_names

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w, language):
    phrase_match = f"{w.replace(' ', '_')}".lower()
    available_signs = get_words_in_database(language)
    if phrase_match in available_signs:
        return phrase_match
    best_score = -1.0
    best_vid_name = None
    for v in available_signs:
        s = similar(w, v)
        if s > best_score:
            best_score = s
            best_vid_name = v
    if best_score >= SIMILARITY_RATIO:
        return best_vid_name
    return None

def spell_word(word, language):
    available_signs = get_words_in_database(language)
    spelled = []
    for ch in word:
        if ch in available_signs:
            spelled.append(ch)
        else:
            return None
    return spelled

def merge_signs(sign_sequence, language, output_path):
    """Merge multiple signs into one video with larger frame"""
    clips = []
    signs_path = get_signs_path(language)
    for sign in sign_sequence:
        sign_path = os.path.join(signs_path, f"{sign}.mp4")
        if os.path.exists(sign_path):
            try:
                clip = VideoFileClip(sign_path, audio=False)
                # Increase frame size significantly - much larger and wider
                clip = clip.resized(height=500)  # Increased to 500px for better visibility
                clip = clip.without_audio()
                clips.append(clip)
            except Exception as e:
                print(f"❌ Error loading sign {sign}: {e}")
                return False
        else:
            print(f"❌ Sign not found: {sign_path}")
            return False
    if clips:
        try:
            final_clip = concatenate_videoclips(clips, method="compose")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use higher quality settings for better output
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio=False, 
                fps=24,
                bitrate="3000k"  # Higher bitrate for better quality
            )
            for clip in clips:
                clip.close()
            final_clip.close()
            return True
        except Exception as e:
            print(f"❌ Error merging signs: {e}")
            return False
    return False

def text_to_sign(text, language, output_path):
    sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    phrase_mappings = {
        "what is your name": "what_is_your_name",
        "how are you": "how_are_you",
        "im great": "im_great",
        "my name is": "my_name_is",
        "good morning": "goodmorning"
    }
    if sanitized_text in phrase_mappings:
        return merge_signs([phrase_mappings[sanitized_text]], language, output_path)
    words = sanitized_text.split()
    final_sequence = []
    for w in words:
        db_match = find_in_db(w, language)
        if db_match:
            final_sequence.append(db_match)
        else:
            spelled = spell_word(w, language)
            if spelled:
                final_sequence.extend(spelled)
    if final_sequence:
        return merge_signs(final_sequence, language, output_path)
    return False

# ==================== INDIVIDUAL SIGN TO TEXT ====================

def init_individual_recognition(language="asl"):
    """Initialize INDIVIDUAL sign to text recognition"""
    global individual_recognizer, individual_camera, is_individual_running, individual_language
    
    print(f"🔧 [INDIVIDUAL] Initializing {language.upper()} sign recognition...")
    individual_language = language
    
    try:
        if individual_camera is not None:
            print("🔄 [INDIVIDUAL] Releasing existing camera...")
            try:
                individual_camera.release()
            except:
                pass
            time.sleep(1)
            individual_camera = None
        
        if individual_recognizer is not None:
            try:
                del individual_recognizer
            except:
                pass
            individual_recognizer = None
        
        from gesture_recognition import GestureRecognizer
        individual_recognizer = GestureRecognizer(language=language)
        
        print("📷 [INDIVIDUAL] Opening camera...")
        individual_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not individual_camera.isOpened():
            print("❌ [INDIVIDUAL] Failed with DirectShow, trying default...")
            individual_camera = cv2.VideoCapture(0)
            
        if not individual_camera.isOpened():
            print("❌ [INDIVIDUAL] Failed to open camera")
            return False
            
        individual_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        individual_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        individual_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, frame = individual_camera.read()
        if not ret:
            print("❌ [INDIVIDUAL] Failed to read from camera")
            individual_camera.release()
            individual_camera = None
            return False
        
        is_individual_running = True
        print(f"✅ [INDIVIDUAL] {language.upper()} recognition initialized")
        
        global individual_sentence, last_individual_gesture
        individual_sentence = ""
        last_individual_gesture = ""
        
        thread = threading.Thread(target=process_individual_frames, daemon=True)
        thread.start()
        
        return True
        
    except Exception as e:
        print(f"❌ [INDIVIDUAL] Failed to initialize: {e}")
        if individual_camera:
            individual_camera.release()
        individual_camera = None
        is_individual_running = False
        return False

def process_individual_frames():
    """Process frames for INDIVIDUAL sign to text"""
    global individual_frame, individual_gesture, individual_sentence, individual_language
    global last_individual_time, last_individual_gesture
    
    print(f"🎥 [INDIVIDUAL] Starting frame processing...")
    
    while is_individual_running:
        try:
            if not individual_camera or not individual_camera.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = individual_camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            if individual_recognizer:
                try:
                    processed_frame, detected_gesture = individual_recognizer.recognize_gesture(frame)
                    individual_gesture = detected_gesture
                    
                    current_time = time.time()
                    if (current_time - last_individual_time > 2.0 and
                        detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
                        individual_sentence += detected_gesture + " "
                        last_individual_gesture = detected_gesture
                        last_individual_time = current_time
                        print(f"📝 [INDIVIDUAL] Added: {detected_gesture}")
                    
                    height, width = processed_frame.shape[:2]
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
                    cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
                    processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
                    cv2.putText(processed_frame, f"Mode: Individual Sign to Text", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Language: {individual_language.upper()}", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Gesture: {individual_gesture}", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Text: {individual_sentence.strip()}", 
                               (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    with individual_frame_lock:
                        individual_frame = processed_frame.copy()
                        
                except Exception as e:
                    print(f"❌ [INDIVIDUAL] Processing error: {e}")
                    with individual_frame_lock:
                        individual_frame = frame.copy()
            else:
                with individual_frame_lock:
                    individual_frame = frame.copy()
            
        except Exception as e:
            print(f"❌ [INDIVIDUAL] Frame error: {e}")
            time.sleep(0.1)
    
    print("🛑 [INDIVIDUAL] Frame processing stopped")

def generate_individual_frames():
    """Generate frames for INDIVIDUAL streaming"""
    global is_individual_streaming
    
    if is_individual_streaming:
        print("⚠️ [INDIVIDUAL] Already streaming")
        return
    
    is_individual_streaming = True
    print("📡 [INDIVIDUAL] Starting stream...")
    
    try:
        while is_individual_running:
            with individual_frame_lock:
                if individual_frame is None:
                    time.sleep(0.1)
                    continue
                frame = individual_frame.copy()
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)
    except Exception as e:
        print(f"❌ [INDIVIDUAL] Stream error: {e}")
    finally:
        is_individual_streaming = False
        print("📡 [INDIVIDUAL] Stream ended")

def stop_individual_recognition():
    """Stop INDIVIDUAL recognition"""
    global is_individual_running, individual_camera, individual_frame, individual_recognizer
    
    print("🔄 [INDIVIDUAL] Stopping recognition...")
    is_individual_running = False
    time.sleep(1)
    
    if individual_camera:
        try:
            individual_camera.release()
            print("✅ [INDIVIDUAL] Camera released")
        except Exception as e:
            print(f"❌ [INDIVIDUAL] Error releasing camera: {e}")
        individual_camera = None
    
    if individual_recognizer:
        try:
            del individual_recognizer
            print("✅ [INDIVIDUAL] Recognizer cleaned up")
        except Exception as e:
            print(f"❌ [INDIVIDUAL] Error cleaning up: {e}")
        individual_recognizer = None
    
    individual_frame = None
    cv2.destroyAllWindows()
    print("✅ [INDIVIDUAL] Stopped")

# ==================== BIDIRECTIONAL TRANSLATION ====================

def translate_gesture_to_language(gesture_text, source_lang, target_lang):
    """Translate gesture between sign languages"""
    if not gesture_text or gesture_text in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]:
        return None
    
    gesture_text_upper = gesture_text.upper()
    
    # For alphabets and common signs
    if len(gesture_text_upper) == 1 and gesture_text_upper.isalpha():
        return gesture_text_upper.lower()
    
    # ISL specific mappings
    if target_lang == "isl":
        isl_mappings = {
            "hello": "namaste",
            "thank you": "dhanyavad",
            "please": "kripaya",
            "yes": "haan",
            "no": "nahi",
            "sorry": "ksama",
            "help": "madad"
        }
        if gesture_text.lower() in isl_mappings:
            return isl_mappings[gesture_text.lower()]
    
    return gesture_text.lower()

def create_combined_video(gesture_sequence, target_language):
    """Create a combined video from multiple gestures - REPLACE previous"""
    if not gesture_sequence:
        return None
    
    # Use fixed filename to REPLACE previous video
    video_filename = "latest_combined.mp4"
    video_path = os.path.join("static", video_filename)
    
    # Remove previous video if exists (this ensures replacement)
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"🗑️ Removed previous combined video to make space for new one")
        except Exception as e:
            print(f"⚠️ Could not remove previous video: {e}")
    
    success = merge_signs(gesture_sequence, target_language, video_path)
    
    if success:
        return {
            'filename': video_filename,
            'path': video_path,
            'timestamp': int(time.time() * 1000),
            'gestures': gesture_sequence,
            'count': len(gesture_sequence),
            'source_lang': source_language,
            'target_lang': target_language
        }
    return None

def init_bidirectional_recognition(source_lang="asl", target_lang="bsl"):
    """Initialize BIDIRECTIONAL translation"""
    global bidirectional_recognizer, bidirectional_camera, is_bidirectional_running
    global source_language, target_language, gesture_buffer
    
    print(f"🔧 [BIDIRECTIONAL] Initializing: {source_lang.upper()} → {target_lang.upper()}...")
    source_language = source_lang
    target_language = target_lang
    
    try:
        if bidirectional_camera is not None:
            print("🔄 [BIDIRECTIONAL] Releasing existing camera...")
            try:
                bidirectional_camera.release()
            except:
                pass
            time.sleep(1)
            bidirectional_camera = None
        
        if bidirectional_recognizer is not None:
            try:
                del bidirectional_recognizer
            except:
                pass
            bidirectional_recognizer = None
        
        from gesture_recognition import GestureRecognizer
        bidirectional_recognizer = GestureRecognizer(language=source_lang)
        
        print("📷 [BIDIRECTIONAL] Opening camera...")
        bidirectional_camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not bidirectional_camera.isOpened():
            print("❌ [BIDIRECTIONAL] Failed with DirectShow, trying default...")
            bidirectional_camera = cv2.VideoCapture(0)
            
        if not bidirectional_camera.isOpened():
            print("❌ [BIDIRECTIONAL] Failed to open camera")
            return False
            
        bidirectional_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        bidirectional_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        bidirectional_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, frame = bidirectional_camera.read()
        if not ret:
            print("❌ [BIDIRECTIONAL] Failed to read from camera")
            bidirectional_camera.release()
            bidirectional_camera = None
            return False
        
        is_bidirectional_running = True
        print(f"✅ [BIDIRECTIONAL] Translation initialized")
        
        global bidirectional_text, translated_video_queue, current_translated_video, gesture_buffer
        bidirectional_text = ""
        translated_video_queue = []
        current_translated_video = None
        gesture_buffer = []  # Reset gesture buffer
        
        thread = threading.Thread(target=process_bidirectional_frames, daemon=True)
        thread.start()
        
        return True
        
    except Exception as e:
        print(f"❌ [BIDIRECTIONAL] Failed to initialize: {e}")
        if bidirectional_camera:
            bidirectional_camera.release()
        bidirectional_camera = None
        is_bidirectional_running = False
        return False

def process_bidirectional_frames():
    """Process frames for BIDIRECTIONAL translation"""
    global bidirectional_frame, bidirectional_gesture, bidirectional_text
    global last_bidirectional_time, translated_video_queue, current_translated_video, gesture_buffer
    
    print(f"🎥 [BIDIRECTIONAL] Starting processing: {source_language.upper()} → {target_language.upper()}...")
    
    while is_bidirectional_running:
        try:
            if not bidirectional_camera or not bidirectional_camera.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = bidirectional_camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            
            if bidirectional_recognizer:
                try:
                    processed_frame, detected_gesture = bidirectional_recognizer.recognize_gesture(frame)
                    bidirectional_gesture = detected_gesture
                    
                    current_time = time.time()
                    if (current_time - last_bidirectional_time > 2.0 and
                        detected_gesture not in ["No hand detected", "Unknown", "Hand detected", "Error", "Analyzing...", "No gesture detected"]):
                        
                        bidirectional_text += detected_gesture + " "
                        last_bidirectional_time = current_time
                        
                        # Translate the gesture
                        translated_gesture = translate_gesture_to_language(
                            detected_gesture, 
                            source_language, 
                            target_language
                        )
                        
                        if translated_gesture:
                            # Add to gesture buffer
                            gesture_buffer.append(translated_gesture)
                            print(f"📝 [BIDIRECTIONAL] Added to buffer: {detected_gesture} → {translated_gesture} (Buffer: {len(gesture_buffer)}/{COMBINE_AFTER_GESTURES})")
                            
                            # Check if we have enough gestures to combine
                            if len(gesture_buffer) >= COMBINE_AFTER_GESTURES:
                                print(f"🎬 [BIDIRECTIONAL] Creating LARGE combined video with {len(gesture_buffer)} gestures...")
                                
                                # Create combined video (will REPLACE previous)
                                combined_video = create_combined_video(gesture_buffer, target_language)
                                
                                if combined_video:
                                    with video_queue_lock:
                                        # REPLACE the queue with only the latest video
                                        translated_video_queue = [combined_video]
                                        current_translated_video = combined_video
                                    
                                    print(f"✅ [BIDIRECTIONAL] Created LARGE combined video: {combined_video['filename']}")
                                    print(f"   ↳ REPLACED previous video with new one")
                                    print(f"   ↳ Gestures: {', '.join(gesture_buffer)}")
                                    
                                    # Clear buffer after creating combined video
                                    gesture_buffer = []
                                else:
                                    print(f"❌ [BIDIRECTIONAL] Failed to create combined video")
                    
                    # Display buffer status on frame
                    height, width = processed_frame.shape[:2]
                    overlay = processed_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (width - 10, 160), (0, 0, 0), -1)
                    cv2.rectangle(overlay, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
                    processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
                    
                    cv2.putText(processed_frame, f"Mode: Bidirectional Translation", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, f"Input: {source_language.upper()}", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Output: {target_language.upper()}", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(processed_frame, f"Gesture: {bidirectional_gesture}", 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Buffer: {len(gesture_buffer)}/{COMBINE_AFTER_GESTURES}", 
                               (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(processed_frame, "Hold gesture for 2 seconds", 
                               (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    with bidirectional_frame_lock:
                        bidirectional_frame = processed_frame.copy()
                        
                except Exception as e:
                    print(f"❌ [BIDIRECTIONAL] Processing error: {e}")
                    with bidirectional_frame_lock:
                        bidirectional_frame = frame.copy()
            else:
                with bidirectional_frame_lock:
                    bidirectional_frame = frame.copy()
            
        except Exception as e:
            print(f"❌ [BIDIRECTIONAL] Frame error: {e}")
            time.sleep(0.1)
    
    print("🛑 [BIDIRECTIONAL] Processing stopped")

def generate_bidirectional_frames():
    """Generate frames for BIDIRECTIONAL streaming"""
    global is_bidirectional_streaming
    
    if is_bidirectional_streaming:
        print("⚠️ [BIDIRECTIONAL] Already streaming")
        return
    
    is_bidirectional_streaming = True
    print("📡 [BIDIRECTIONAL] Starting stream...")
    
    try:
        while is_bidirectional_running:
            with bidirectional_frame_lock:
                if bidirectional_frame is None:
                    time.sleep(0.1)
                    continue
                frame = bidirectional_frame.copy()
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)
    except Exception as e:
        print(f"❌ [BIDIRECTIONAL] Stream error: {e}")
    finally:
        is_bidirectional_streaming = False
        print("📡 [BIDIRECTIONAL] Stream ended")

def stop_bidirectional_recognition():
    """Stop BIDIRECTIONAL translation"""
    global is_bidirectional_running, bidirectional_camera, bidirectional_frame, bidirectional_recognizer
    global gesture_buffer
    
    print("🔄 [BIDIRECTIONAL] Stopping...")
    is_bidirectional_running = False
    time.sleep(1)
    
    if bidirectional_camera:
        try:
            bidirectional_camera.release()
            print("✅ [BIDIRECTIONAL] Camera released")
        except Exception as e:
            print(f"❌ [BIDIRECTIONAL] Error releasing camera: {e}")
        bidirectional_camera = None
    
    if bidirectional_recognizer:
        try:
            del bidirectional_recognizer
            print("✅ [BIDIRECTIONAL] Recognizer cleaned up")
        except Exception as e:
            print(f"❌ [BIDIRECTIONAL] Error cleaning up: {e}")
        bidirectional_recognizer = None
    
    bidirectional_frame = None
    gesture_buffer = []  # Clear buffer
    cv2.destroyAllWindows()
    print("✅ [BIDIRECTIONAL] Stopped")

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

# ===== TEXT TO SIGN ROUTES =====
@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'asl')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        if language not in SIGN_LANGUAGES:
            language = DEFAULT_LANGUAGE
        
        # Use fixed filename for text-to-sign - REPLACES previous
        output_path = os.path.join("static", "latest_text_to_sign.mp4")
        
        # Remove previous video if exists (ensures replacement)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"🗑️ Removed previous text-to-sign video to make space for new one")
            except Exception as e:
                print(f"⚠️ Could not remove previous video: {e}")
        
        success = text_to_sign(text, language, output_path)
        
        if success and os.path.exists(output_path):
            return jsonify({
                'success': True,
                'message': f'LARGE video created for: {text} (REPLACED previous)',
                'video_url': f'/video/text_to_sign?t={int(time.time() * 1000)}'  # Cache busting
            })
        else:
            return jsonify({'success': False, 'error': 'Could not generate video'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# New route for text-to-sign video
@app.route('/video/text_to_sign')
def serve_text_to_sign_video():
    video_path = os.path.join("static", "latest_text_to_sign.mp4")
    if os.path.exists(video_path):
        print(f"🎬 Serving text-to-sign video: {video_path}")
        return send_file(video_path, as_attachment=False, mimetype='video/mp4')
    else:
        print(f"❌ Text-to-sign video not found: {video_path}")
        return jsonify({'error': 'Video not found'}), 404

# Updated route for bidirectional video
@app.route('/video/bidirectional')
def serve_bidirectional_video():
    video_path = os.path.join("static", "latest_combined.mp4")
    if os.path.exists(video_path):
        print(f"🎬 Serving bidirectional video: {video_path}")
        return send_file(video_path, as_attachment=False, mimetype='video/mp4')
    else:
        print(f"❌ Bidirectional video not found: {video_path}")
        return jsonify({'error': 'Video not found'}), 404

# ===== INDIVIDUAL SIGN TO TEXT ROUTES =====
@app.route('/start_individual')
def start_individual():
    global individual_sentence, last_individual_gesture
    try:
        language = request.args.get('language', 'asl')
        if language not in ['asl', 'bsl', 'isl']:
            language = 'asl'
            
        print(f"🔄 [INDIVIDUAL] Starting {language.upper()}...")
        
        stop_individual_recognition()
        time.sleep(2)
        
        success = init_individual_recognition(language)
        if success:
            individual_sentence = ""
            last_individual_gesture = ""
            return jsonify({
                'success': True, 
                'message': f'{language.upper()} Individual recognition started', 
                'language': language.upper()
            })
        else:
            return jsonify({'success': False, 'error': f'Failed to start {language.upper()}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_individual')
def stop_individual():
    stop_individual_recognition()
    return jsonify({'success': True, 'message': 'Individual recognition stopped'})

@app.route('/individual_feed')
def individual_feed():
    if not is_individual_running:
        return jsonify({'error': 'Individual recognition not running'}), 400
    return Response(generate_individual_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_individual_data')
def get_individual_data():
    return jsonify({
        'current_gesture': individual_gesture,
        'sentence': individual_sentence.strip(),
        'running': is_individual_running,
        'language': individual_language.upper()
    })

@app.route('/clear_individual')
def clear_individual():
    global individual_sentence, last_individual_gesture
    individual_sentence = ""
    last_individual_gesture = ""
    return jsonify({'success': True, 'message': 'Text cleared'})

# ===== BIDIRECTIONAL TRANSLATION ROUTES =====
@app.route('/start_bidirectional')
def start_bidirectional():
    global bidirectional_text, translated_video_queue, current_translated_video, gesture_buffer
    try:
        source = request.args.get('source', 'asl')
        target = request.args.get('target', 'bsl')
        
        if source not in ['asl', 'bsl', 'isl']:
            source = 'asl'
        if target not in ['asl', 'bsl', 'isl']:
            target = 'bsl'
            
        print(f"🔄 [BIDIRECTIONAL] Starting: {source.upper()} → {target.upper()}...")
        
        stop_bidirectional_recognition()
        time.sleep(2)
        
        success = init_bidirectional_recognition(source, target)
        if success:
            bidirectional_text = ""
            translated_video_queue = []
            current_translated_video = None
            gesture_buffer = []
            return jsonify({
                'success': True, 
                'message': f'Translation started: {source.upper()} → {target.upper()}',
                'source': source.upper(),
                'target': target.upper()
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to start translation'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_bidirectional')
def stop_bidirectional():
    stop_bidirectional_recognition()
    return jsonify({'success': True, 'message': 'Translation stopped'})

@app.route('/bidirectional_feed')
def bidirectional_feed():
    if not is_bidirectional_running:
        return jsonify({'error': 'Translation not running'}), 400
    return Response(generate_bidirectional_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_bidirectional_data')
def get_bidirectional_data():
    with video_queue_lock:
        latest_video = current_translated_video.copy() if current_translated_video else None
    
    # Check if latest combined video exists
    latest_video_path = os.path.join("static", "latest_combined.mp4")
    has_latest_video = os.path.exists(latest_video_path)
    
    return jsonify({
        'current_gesture': bidirectional_gesture,
        'text': bidirectional_text.strip(),
        'running': is_bidirectional_running,
        'source_language': source_language.upper(),
        'target_language': target_language.upper(),
        'latest_video': latest_video,
        'has_combined_video': has_latest_video,
        'video_count': len(translated_video_queue),
        'buffer_count': len(gesture_buffer),
        'buffer_max': COMBINE_AFTER_GESTURES
    })

@app.route('/clear_bidirectional')
def clear_bidirectional():
    global bidirectional_text, gesture_buffer
    bidirectional_text = ""
    gesture_buffer = []  # Clear buffer when clearing text
    return jsonify({'success': True, 'message': 'Translation text and buffer cleared'})

@app.route('/get_translated_videos')
def get_translated_videos():
    """Get list of all translated videos"""
    with video_queue_lock:
        videos = translated_video_queue.copy()
    
    # Check if latest combined video exists
    latest_video_path = os.path.join("static", "latest_combined.mp4")
    has_latest_video = os.path.exists(latest_video_path)
    
    return jsonify({
        'success': True,
        'videos': videos,
        'has_latest_video': has_latest_video,
        'buffer_count': len(gesture_buffer),
        'buffer_max': COMBINE_AFTER_GESTURES
    })

# ===== DEBUG ROUTE =====
@app.route('/debug')
def debug_info():
    # Check if latest videos exist
    text_to_sign_exists = os.path.exists(os.path.join("static", "latest_text_to_sign.mp4"))
    bidirectional_exists = os.path.exists(os.path.join("static", "latest_combined.mp4"))
    
    return jsonify({
        'text_to_sign': {
            'languages': list(SIGN_LANGUAGES.keys()),
            'has_latest_video': text_to_sign_exists,
            'video_path': 'latest_text_to_sign.mp4'
        },
        'individual': {
            'language': individual_language,
            'status': 'running' if is_individual_running else 'stopped',
            'sentence': individual_sentence,
            'streaming': is_individual_streaming
        },
        'bidirectional': {
            'source': source_language,
            'target': target_language,
            'status': 'running' if is_bidirectional_running else 'stopped',
            'text': bidirectional_text,
            'streaming': is_bidirectional_streaming,
            'video_count': len(translated_video_queue),
            'has_latest_video': bidirectional_exists,
            'video_path': 'latest_combined.mp4',
            'buffer_count': len(gesture_buffer),
            'buffer_max': COMBINE_AFTER_GESTURES
        }
    })

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    
    print("🚀 Starting Complete Sign Language Translation App...")
    print("\n📁 Available Sign Languages:")
    for lang in SIGN_LANGUAGES:
        signs = get_words_in_database(lang)
        print(f"   - {lang.upper()}: {len(signs)} signs")
    
    print(f"\n🔄 Bidirectional Translation: Combines videos every {COMBINE_AFTER_GESTURES} gestures")
    print("🎬 Video Features:")
    print("   - EXTRA LARGE output frames (500px height)")
    print("   - Latest videos REPLACE previous ones")
    print("   - High quality encoding (3000k bitrate)")
    print("\n🌐 Server starting on http://localhost:5001")
    print("\n✨ Features:")
    print("   1️⃣  Text to Sign - Convert text to LARGE sign language videos")
    print("   2️⃣  Individual Sign to Text - Simple sign recognition → text")
    print("   3️⃣  Bidirectional Translation - Sign in one language → COMBINED video in another")
    print("\n   Supported Languages: ASL, BSL, ISL (with 2-hand detection for BSL & ISL)")
    print("\n" + "="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)