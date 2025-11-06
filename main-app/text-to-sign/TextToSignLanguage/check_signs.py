import os

SIGN_PATH = r"C:\Users\laasy\OneDrive\Desktop\cst12\TextToSignLanguage\signs"

files = os.listdir(SIGN_PATH)
print("Files in signs folder:", files)

videos = [os.path.splitext(f)[0].lower() for f in files if f.endswith(".mp4")]
print("Detected video names:", videos)
