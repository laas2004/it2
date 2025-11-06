import os
import nltk
import re    #added for v3 of main
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from difflib import SequenceMatcher
from moviepy import VideoFileClip, concatenate_videoclips

# Make sure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# CONSTANTS
SIGN_PATH = "signs"
OUTPUT_PATH = os.path.join(SIGN_PATH, "Output", "out.mp4")
SIMILARITY_RATIO = 0.9

# -------------------- Helper Functions --------------------

def get_words_in_database():
    """List all available signs (without .mp4 extension)."""
    vids = [f for f in os.listdir(SIGN_PATH) if f.endswith(".mp4")]
    vid_names = [v[:-4].lower() for v in vids]
    return vid_names

def similar(a, b):
    """Return similarity score between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def find_in_db(w):
    """Find the closest match for a word or phrase in the signs database."""
    # Try to find a direct match for the phrase (e.g., 'im_great.mp4')
    phrase_match = f"{w.replace(' ', '_')}".lower()  # Convert 'im great' to 'im_great'
    if phrase_match in get_words_in_database():
        return phrase_match
    # If no direct match, fallback to individual word matching
    best_score = -1.0
    best_vid_name = None
    for v in get_words_in_database():
        s = similar(w, v)
        if s > best_score:
            best_score = s
            best_vid_name = v
    if best_score >= SIMILARITY_RATIO:
        return best_vid_name
    return None

def process_text(text):
    """Tokenize and clean text into a list of words."""
    words = word_tokenize(text)
    useful_words = [str(w).lower() for w in words if w.isalpha()]
    return useful_words

def spell_word(word):
    """Spell a word letter by letter if full sign not available."""
    available_signs = get_words_in_database()
    spelled = []
    for ch in word:
        if ch in available_signs:
            spelled.append(ch)
        else:
            print(f"⚠️ Missing letter: {ch}, cannot fully spell '{word}'")
            return None
    return spelled

def merge_signs(sign_sequence, output_path=OUTPUT_PATH):
    """Concatenate video clips using MoviePy and export final video."""
    clips = []
    for sign in sign_sequence:
        sign_path = os.path.join(SIGN_PATH, f"{sign}.mp4")
        if os.path.exists(sign_path):
            clip = VideoFileClip(sign_path, audio=False).resized(height=180)


            # ✅ Standardize resolution (e.g., 320x240)
            clip = clip.resized(height=240)

            # ✅ Remove audio track if any
            clip = clip.without_audio()

            clips.append(clip)
        else:
            print(f"⚠️ Missing file: {sign_path}, skipping...")

    if clips:
        final_clip = concatenate_videoclips(clips, method="compose")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_clip.write_videofile(output_path, codec="libx264", audio=False)
        print(f"✅ Output saved to: {output_path}")
    else:
        print("❌ No clips to merge.")


# -------------------- Main --------------------

#v1 of main

# if __name__ == "__main__":
#     input_text = input("Enter the text you want to convert: ").strip()

#     words = input_text.lower().split()
#     final_sequence = []

#     for w in words:
#         db_match = find_in_db(w)
#         if db_match:
#             print(f"{w} found in database as '{db_match}'")
#             final_sequence.append(db_match)
#         else:
#             print(f"{w} not found in database, trying to spell it...")
#             spelled = spell_word(w)
#             if spelled:
#                 print(f"{w} → spelling as: {spelled}")
#                 final_sequence.extend(spelled)
#             else:
#                 print(f"❌ Cannot represent {w}, skipping...")

#     if final_sequence:
#         merge_signs(final_sequence)
#     else:
#         print("❌ No matching or spellable words found. Add more sign videos to 'signs' folder.")


#v2 of main  (lets you type and get a phrase)
# if __name__ == "__main__":
#     input_text = input("Enter the text you want to convert: ").strip().lower()

#     # 🔁 Phrase-level support
#     phrase_to_video = {
#         "what is your name": "signs/what_is_your_name.mp4",  # Adjust path if needed
#         # Add more phrases here
#     }

#     if input_text in phrase_to_video:
#         phrase_video_path = phrase_to_video[input_text]
#         if os.path.exists(phrase_video_path):
#             print(f"✅ Found phrase video for '{input_text}'.")
#             os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#             clip = VideoFileClip(phrase_video_path, audio=False).resized(height=240).without_audio()
#             clip.write_videofile(OUTPUT_PATH, codec="libx264", audio=False)
#             print(f"✅ Output saved to: {OUTPUT_PATH}")
#         else:
#             print(f"⚠️ Phrase video file not found: {phrase_video_path}")
#     else:
#         # Fallback to word-by-word processing
#         words = input_text.split()
#         final_sequence = []

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
#             merge_signs(final_sequence)
#         else:
#             print("❌ No matching or spellable words found. Add more sign videos to 'signs' folder.")


#v3 of main(to deal with apostrophes in text)
if __name__ == "__main__":
    input_text = input("Enter the text you want to convert: ").strip()

    # Sanitize the input (remove apostrophes and special characters)
    sanitized_text = re.sub(r"[^a-zA-Z\s]", "", input_text).lower()

    print(f"Sanitized Input: {sanitized_text}")  

    # Phrase-level support 
    phrase_to_video = {
        "what is your name": "signs/what_is_your_name.mp4", 
        # To add more phrases here
    }

    if sanitized_text in phrase_to_video:
        phrase_video_path = phrase_to_video[sanitized_text]
        if os.path.exists(phrase_video_path):
            print(f"✅ Found phrase video for '{sanitized_text}'.")
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            clip = VideoFileClip(phrase_video_path, audio=False).resized(height=240).without_audio()
            clip.write_videofile(OUTPUT_PATH, codec="libx264", audio=False)
            print(f"✅ Output saved to: {OUTPUT_PATH}")
        else:
            print(f"⚠️ Phrase video file not found: {phrase_video_path}")
    else:
        # word-by-word processing
        words = sanitized_text.split()
        final_sequence = []

        # Trying to match whole phrases first (like "im great")
        for i in range(len(words), 0, -1):  
            phrase = " ".join(words[:i])
            db_match = find_in_db(phrase)
            if db_match:
                print(f"Found phrase: '{phrase}' as '{db_match}'")
                final_sequence.append(db_match)
                words = words[i:]  # Remove the matched phrase from the word list
                break

        # After matching a phrase, process remaining individual words
        for w in words:
            db_match = find_in_db(w)
            if db_match:
                print(f"{w} found in database as '{db_match}'")
                final_sequence.append(db_match)
            else:
                print(f"{w} not found in database, trying to spell it...")
                spelled = spell_word(w)
                if spelled:
                    print(f"{w} → spelling as: {spelled}")
                    final_sequence.extend(spelled)
                else:
                    print(f"❌ Cannot r epresent {w}, skipping...")

        if final_sequence:
            merge_signs(final_sequence)
        else:
            print("❌ No matching or spellable words found. Add more sign videos to 'signs' folder.")


