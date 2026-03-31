import gdown
import os
import json
from tqdm import tqdm
import time
import random
# from moviepy.editor import VideoFileClip

def download_from_google_drive(file_id, destination, retries=3):
    """Download a file from Google Drive using the file ID and save it to the destination path."""
    url = f"https://drive.google.com/uc?id={file_id}"

    for _ in range(retries):
        try:
            gdown.download(url, destination, quiet=False)
            return True
        except Exception as e:
            print(f"[ERROR] Issue downloading video from Google Drive: {e}. Retrying...")
    print("[WARNING] Failed to download after multiple retries.")
    return False


random.seed(8)

if __name__ == "__main__":
    # Load necessary JSON files
    with open("questions.json") as questions_f:
        questions = json.load(questions_f)
    
    filtered_questions = []
    with open('subset_answers.json', 'r') as answers:
        data = json.load(answers)

        for q in questions:
            if q["q_uid"] in data:
                filtered_questions.append(q)
    
    print(f'Size of filtered questions:  {len(filtered_questions)}')
    questions = random.sample(filtered_questions, 250)
    print(f'size of filtered questions: {len(questions)}')

    # Create videos directory if it doesn't exist
    os.makedirs("videos", exist_ok=True)

    # Download videos
    for q in tqdm(questions):
        q_uid = q["q_uid"]
        drive_id = q["google_drive_id"]

        if not os.path.exists(os.path.join("videos", f"{q_uid}.mp4")):
            download_from_google_drive(drive_id, os.path.join("videos", f"{q_uid}.mp4"))
            time.sleep(3)
