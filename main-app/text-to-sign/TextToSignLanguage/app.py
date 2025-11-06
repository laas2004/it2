# from flask import Flask, render_template, request, jsonify
# import sys
# import os

# # Add your existing Python module to path
# sys.path.append(os.path.dirname(__file__))

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '')
        
#         # Call your existing text-to-sign logic here
#         # This will depend on how your signtosignlanguage.py works
#         # Example:
#         # from signtosignlanguage import text_to_sign
#         # sign_output = text_to_sign(text)
        
#         # For now, return mock data
#         sign_output = f"Sign translation for: {text}"
        
#         return jsonify({
#             'success': True,
#             'input': text,
#             'sign_output': sign_output
#         })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)


#trying to make it work  v1
# from flask import Flask, render_template, request, jsonify
# import sys
# import os
# import nltk
# import re
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
# SIGN_PATH = "signs"
# OUTPUT_PATH = os.path.join(SIGN_PATH, "Output", "out.mp4")
# SIMILARITY_RATIO = 0.9

# # -------------------- Helper Functions from signtosignlanguage.py --------------------

# def get_words_in_database():
#     """List all available signs (without .mp4 extension)."""
#     vids = [f for f in os.listdir(SIGN_PATH) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w):
#     """Find the closest match for a word or phrase in the signs database."""
#     # Try to find a direct match for the phrase (e.g., 'im_great.mp4')
#     phrase_match = f"{w.replace(' ', '_')}".lower()  # Convert 'im great' to 'im_great'
#     if phrase_match in get_words_in_database():
#         return phrase_match
#     # If no direct match, fallback to individual word matching
#     best_score = -1.0
#     best_vid_name = None
#     for v in get_words_in_database():
#         s = similar(w, v)
#         if s > best_score:
#             best_score = s
#             best_vid_name = v
#     if best_score >= SIMILARITY_RATIO:
#         return best_vid_name
#     return None

# def spell_word(word):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database()
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
#             return None
#     return spelled

# def merge_signs(sign_sequence, output_path=OUTPUT_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     for sign in sign_sequence:
#         sign_path = os.path.join(SIGN_PATH, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             clip = VideoFileClip(sign_path, audio=False).resized(height=240)
#             clip = clip.without_audio()
#             clips.append(clip)
#         else:
#             print(f"⚠️ Missing file: {sign_path}, skipping...")

#     if clips:
#         final_clip = concatenate_videoclips(clips, method="compose")
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         final_clip.write_videofile(output_path, codec="libx264", audio=False)
#         print(f"✅ Output saved to: {output_path}")
#         return True
#     else:
#         print("❌ No clips to merge.")
#         return False

# def text_to_sign(text):
#     """Main function to convert text to sign language video - v3 logic"""
#     # Sanitize the input (remove apostrophes and special characters)
#     sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    
#     print(f"Sanitized Input: {sanitized_text}")  

#     # Phrase-level support 
#     phrase_to_video = {
#         "what is your name": "signs/what_is_your_name.mp4", 
#         # Add more phrases here
#     }

#     if sanitized_text in phrase_to_video:
#         phrase_video_path = phrase_to_video[sanitized_text]
#         if os.path.exists(phrase_video_path):
#             print(f"✅ Found phrase video for '{sanitized_text}'.")
#             os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#             clip = VideoFileClip(phrase_video_path, audio=False).resized(height=240).without_audio()
#             clip.write_videofile(OUTPUT_PATH, codec="libx264", audio=False)
#             print(f"✅ Output saved to: {OUTPUT_PATH}")
#             return True
#         else:
#             print(f"⚠️ Phrase video file not found: {phrase_video_path}")
#             return False
#     else:
#         # word-by-word processing
#         words = sanitized_text.split()
#         final_sequence = []

#         # Trying to match whole phrases first (like "im great")
#         for i in range(len(words), 0, -1):  
#             phrase = " ".join(words[:i])
#             db_match = find_in_db(phrase)
#             if db_match:
#                 print(f"Found phrase: '{phrase}' as '{db_match}'")
#                 final_sequence.append(db_match)
#                 words = words[i:]  # Remove the matched phrase from the word list
#                 break

#         # After matching a phrase, process remaining individual words
#         for w in words:
#             db_match = find_in_db(w)
#             if db_match:
#                 print(f"{w} found in database as '{db_match}'")
#                 final_sequence.append(db_match)
#             else:
#                 print(f"{w} not found in database, trying to spell it...")
#                 spelled = spell_word(w)
#                 if spelled:
#                     print(f"{w} → spelling as: {spelled}")
#                     final_sequence.extend(spelled)
#                 else:
#                     print(f"❌ Cannot represent {w}, skipping...")

#         if final_sequence:
#             success = merge_signs(final_sequence)
#             return success
#         else:
#             print("❌ No matching or spellable words found. Add more sign videos to 'signs' folder.")
#             return False

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '')
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         # Call your text-to-sign logic here
#         success = text_to_sign(text)
        
#         if success:
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'message': f'Sign language video created for: {text}',
#                 'video_path': OUTPUT_PATH
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Could not generate sign language video'
#             })
            
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def get_video():
#     """Serve the generated video file"""
#     try:
#         if os.path.exists(OUTPUT_PATH):
#             return jsonify({
#                 'success': True,
#                 'video_url': f'/static/signs/Output/out.mp4'
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Video not found'
#             })
#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.join(SIGN_PATH, "Output"), exist_ok=True)
#     app.run(port=5001, debug=True)


#trying to make it work  v2
# from flask import Flask, render_template, request, jsonify, send_file
# import sys
# import os
# import nltk
# import re
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
# SIGN_PATH = os.path.join(BASE_DIR, "signs")
# STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
# SIMILARITY_RATIO = 0.9

# print(f"📁 Signs folder path: {SIGN_PATH}")
# print(f"📁 Static video path: {STATIC_VIDEO_PATH}")

# # -------------------- Helper Functions --------------------

# def get_words_in_database():
#     """List all available signs (without .mp4 extension)."""
#     if not os.path.exists(SIGN_PATH):
#         print(f"❌ Signs directory does not exist: {SIGN_PATH}")
#         return []
    
#     vids = [f for f in os.listdir(SIGN_PATH) if f.endswith(".mp4")]
#     vid_names = [v[:-4].lower() for v in vids]
#     return vid_names

# def similar(a, b):
#     """Return similarity score between two strings."""
#     return SequenceMatcher(None, a, b).ratio()

# def find_in_db(w):
#     """Find the closest match for a word or phrase in the signs database."""
#     # Try to find a direct match for the phrase
#     phrase_match = f"{w.replace(' ', '_')}".lower()
#     available_signs = get_words_in_database()
    
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

# def spell_word(word):
#     """Spell a word letter by letter if full sign not available."""
#     available_signs = get_words_in_database()
#     spelled = []
#     for ch in word:
#         if ch in available_signs:
#             spelled.append(ch)
#         else:
#             print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
#             return None
#     return spelled

# def merge_signs(sign_sequence, output_path=STATIC_VIDEO_PATH):
#     """Concatenate video clips using MoviePy and export final video."""
#     clips = []
#     for sign in sign_sequence:
#         sign_path = os.path.join(SIGN_PATH, f"{sign}.mp4")
#         if os.path.exists(sign_path):
#             try:
#                 clip = VideoFileClip(sign_path, audio=False)
#                 # Resize to consistent size
#                 clip = clip.resized(height=240)
#                 clip = clip.without_audio()
#                 clips.append(clip)
#                 print(f"✅ Loaded: {sign}")
#             except Exception as e:
#                 print(f"❌ Error loading clip {sign_path}: {e}")
#                 return False
#         else:
#             print(f"⚠️ Missing file: {sign_path}")
#             return False

#     if clips:
#         try:
#             print(f"🎞️ Merging {len(clips)} clips...")
#             final_clip = concatenate_videoclips(clips, method="compose")
#             # Ensure output directory exists
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             # Write the video file - simplified for newer MoviePy versions
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

# def text_to_sign(text):
#     """Main function to convert text to sign language video"""
#     print(f"🎯 Starting translation for: '{text}'")
    
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
#         success = merge_signs([phrase_sign])
#         return success
    
#     # Word-by-word processing for other text
#     words = sanitized_text.split()
#     final_sequence = []

#     # Process each word
#     for w in words:
#         db_match = find_in_db(w)
#         if db_match:
#             print(f"✅ '{w}' found in database as '{db_match}'")
#             final_sequence.append(db_match)
#         else:
#             print(f"❌ '{w}' not found in database, trying to spell it...")
#             spelled = spell_word(w)
#             if spelled:
#                 print(f"🔤 '{w}' → spelling as: {spelled}")
#                 final_sequence.extend(spelled)
#             else:
#                 print(f"❌ Cannot represent '{w}', skipping...")

#     print(f"📋 Final sequence: {final_sequence}")
    
#     if final_sequence:
#         success = merge_signs(final_sequence)
#         return success
#     else:
#         print("❌ No matching or spellable words found.")
#         return False

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/translate', methods=['POST'])
# def translate_text():
#     try:
#         data = request.json
#         text = data.get('text', '').strip()
        
#         if not text:
#             return jsonify({
#                 'success': False,
#                 'error': 'No text provided'
#             })
        
#         print(f"📥 Received text to translate: '{text}'")
        
#         # Call your text-to-sign logic here
#         success = text_to_sign(text)
        
#         if success and os.path.exists(STATIC_VIDEO_PATH):
#             return jsonify({
#                 'success': True,
#                 'input': text,
#                 'message': f'Sign language video created for: {text}',
#                 'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'  # Cache busting
#             })
#         else:
#             return jsonify({
#                 'success': False,
#                 'error': 'Could not generate sign language video. Please try a different word.'
#             })
            
#     except Exception as e:
#         print(f"❌ Route error: {e}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })

# @app.route('/video')
# def serve_video():
#     """Serve the generated video file"""
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

# @app.route('/debug')
# def debug():
#     """Debug endpoint to check paths and available signs"""
#     debug_info = {
#         'base_dir': BASE_DIR,
#         'signs_path': SIGN_PATH,
#         'signs_exists': os.path.exists(SIGN_PATH),
#         'available_signs': get_words_in_database(),
#         'static_video_exists': os.path.exists(STATIC_VIDEO_PATH),
#         'current_working_dir': os.getcwd()
#     }
#     return jsonify(debug_info)

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
#     print("🚀 Starting Flask app...")
#     print(f"📁 Current working directory: {os.getcwd()}")
#     print(f"📁 Signs path: {SIGN_PATH}")
#     print(f"📁 Signs exists: {os.path.exists(SIGN_PATH)}")
#     print(f"📁 Available signs: {get_words_in_database()}")
    
#     app.run(port=5001, debug=True)




#adding asl,bsl,isl v3
from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
import nltk
import re
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from moviepy import VideoFileClip, concatenate_videoclips

# Add your existing Python module to path
sys.path.append(os.path.dirname(__file__))

app = Flask(__name__)

# Make sure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# CONSTANTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGN_LANGUAGES = {
    "asl": "signs",           # American Sign Language
    "bsl": "signs_bsl",       # British Sign Language  
    "isl": "signs_isl"        # Indian Sign Language
}
DEFAULT_LANGUAGE = "asl"
STATIC_VIDEO_PATH = os.path.join("static", "output_video.mp4")
SIMILARITY_RATIO = 0.9

print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")

# -------------------- Helper Functions --------------------

def get_signs_path(language):
    """Get the path for the selected sign language."""
    folder_name = SIGN_LANGUAGES.get(language, SIGN_LANGUAGES[DEFAULT_LANGUAGE])
    return os.path.join(BASE_DIR, folder_name)

def get_words_in_database(language):
    """List all available signs for the selected language."""
    signs_path = get_signs_path(language)
    print(f"🔍 Looking for signs in: {signs_path} for language: {language}")
    
    if not os.path.exists(signs_path):
        print(f"❌ Signs directory does not exist: {signs_path}")
        return []
    
    vids = [f for f in os.listdir(signs_path) if f.endswith(".mp4")]
    vid_names = [v[:-4].lower() for v in vids]
    print(f"📹 Found {len(vid_names)} sign videos for {language}: {vid_names}")
    return vid_names

def similar(a, b):
    """Return similarity score between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w, language):
    """Find the closest match for a word or phrase in the signs database."""
    # Try to find a direct match for the phrase
    phrase_match = f"{w.replace(' ', '_')}".lower()
    available_signs = get_words_in_database(language)
    
    if phrase_match in available_signs:
        return phrase_match
    
    # If no direct match, fallback to individual word matching
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
    """Spell a word letter by letter if full sign not available."""
    available_signs = get_words_in_database(language)
    spelled = []
    for ch in word:
        if ch in available_signs:
            spelled.append(ch)
        else:
            print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
            return None
    return spelled

def merge_signs(sign_sequence, language, output_path=STATIC_VIDEO_PATH):
    """Concatenate video clips using MoviePy and export final video."""
    clips = []
    signs_path = get_signs_path(language)
    
    for sign in sign_sequence:
        sign_path = os.path.join(signs_path, f"{sign}.mp4")
        if os.path.exists(sign_path):
            try:
                clip = VideoFileClip(sign_path, audio=False)
                # Resize to consistent size
                clip = clip.resized(height=240)
                clip = clip.without_audio()
                clips.append(clip)
                print(f"✅ Loaded: {sign} for {language}")
            except Exception as e:
                print(f"❌ Error loading clip {sign_path}: {e}")
                return False
        else:
            print(f"⚠️ Missing file: {sign_path}")
            return False

    if clips:
        try:
            print(f"🎞️ Merging {len(clips)} clips for {language}...")
            final_clip = concatenate_videoclips(clips, method="compose")
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Write the video file
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio=False,
                fps=24
            )
            # Close clips to free memory
            for clip in clips:
                clip.close()
            final_clip.close()
            print(f"✅ Output saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error merging clips: {e}")
            return False
    else:
        print("❌ No clips to merge.")
        return False

def text_to_sign(text, language):
    """Main function to convert text to sign language video"""
    print(f"🎯 Starting translation for: '{text}' in {language}")
    
    # Sanitize the input
    sanitized_text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    print(f"🧹 Sanitized Input: '{sanitized_text}'")

    # Check for direct phrase matches first
    phrase_mappings = {
        "what is your name": "what_is_your_name",
        "how are you": "how_are_you", 
        "im great": "im_great",
        "my name is": "my_name_is",
        "good morning": "goodmorning"
    }
    
    if sanitized_text in phrase_mappings:
        phrase_sign = phrase_mappings[sanitized_text]
        print(f"✅ Found phrase mapping: '{sanitized_text}' -> '{phrase_sign}'")
        success = merge_signs([phrase_sign], language)
        return success
    
    # Word-by-word processing for other text
    words = sanitized_text.split()
    final_sequence = []

    # Process each word
    for w in words:
        db_match = find_in_db(w, language)
        if db_match:
            print(f"✅ '{w}' found in database as '{db_match}'")
            final_sequence.append(db_match)
        else:
            print(f"❌ '{w}' not found in database, trying to spell it...")
            spelled = spell_word(w, language)
            if spelled:
                print(f"🔤 '{w}' → spelling as: {spelled}")
                final_sequence.extend(spelled)
            else:
                print(f"❌ Cannot represent '{w}', skipping...")

    print(f"📋 Final sequence for {language}: {final_sequence}")
    
    if final_sequence:
        success = merge_signs(final_sequence, language)
        return success
    else:
        print("❌ No matching or spellable words found.")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'asl')  # Get selected language
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        # Validate language
        if language not in SIGN_LANGUAGES:
            language = DEFAULT_LANGUAGE
            
        print(f"📥 Received text to translate: '{text}' in {language}")
        
        # Call your text-to-sign logic here
        success = text_to_sign(text, language)
        
        if success and os.path.exists(STATIC_VIDEO_PATH):
            return jsonify({
                'success': True,
                'input': text,
                'language': language,
                'message': f'Sign language video created for: {text} ({language.upper()})',
                'video_url': f'/video?t={os.path.getmtime(STATIC_VIDEO_PATH)}'  # Cache busting
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not generate {language.upper()} sign language video. Please try a different word.'
            })
            
    except Exception as e:
        print(f"❌ Route error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/video')
def serve_video():
    """Serve the generated video file"""
    try:
        if os.path.exists(STATIC_VIDEO_PATH):
            return send_file(STATIC_VIDEO_PATH, as_attachment=False, mimetype='video/mp4')
        else:
            return jsonify({
                'success': False,
                'error': 'Video not found'
            }), 404
    except Exception as e:
        print(f"❌ Video serve error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check paths and available signs"""
    debug_info = {
        'base_dir': BASE_DIR,
        'available_languages': list(SIGN_LANGUAGES.keys()),
        'current_working_dir': os.getcwd()
    }
    
    # Add info for each language
    for lang in SIGN_LANGUAGES:
        signs_path = get_signs_path(lang)
        debug_info[f'signs_path_{lang}'] = signs_path
        debug_info[f'signs_exists_{lang}'] = os.path.exists(signs_path)
        debug_info[f'available_signs_{lang}'] = get_words_in_database(lang)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.dirname(STATIC_VIDEO_PATH), exist_ok=True)
    
    print("🚀 Starting Flask app...")
    print(f"📁 Available sign languages: {list(SIGN_LANGUAGES.keys())}")
    
    for lang, folder in SIGN_LANGUAGES.items():
        path = os.path.join(BASE_DIR, folder)
        exists = os.path.exists(path)
        print(f"📁 {lang.upper()} path: {path} - {'✅ Exists' if exists else '❌ Missing'}")
        if exists:
            signs = get_words_in_database(lang)
            print(f"   📹 Available signs: {len(signs)}")
    
    app.run(port=5001, debug=True)