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
