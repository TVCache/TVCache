import json
import os
from pathlib import Path


def load_questions(questions_file):
    with open(questions_file, 'r') as f:
        questions_list = json.load(f)

    questions_dict = {q['q_uid']: q for q in questions_list}
    return questions_dict


def load_answers(answers_file):
    with open(answers_file, 'r') as f:
        answers = json.load(f)
    return answers


def extract_video_id(video_filename):
    return Path(video_filename).stem


def process_videos(videos_dir, questions_file, answers_file):

    print("Loading questions and answers...")
    questions = load_questions(questions_file)
    answers = load_answers(answers_file)

    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos")

    results = []
    missing_questions = []
    missing_answers = []

    for video_file in sorted(video_files):
        video_id = extract_video_id(video_file)

        # Get question
        question_data = questions.get(video_id)
        if question_data is None:
            missing_questions.append(video_id)
            continue

        # Get answer
        answer_idx = answers.get(video_id)
        if answer_idx is None:
            missing_answers.append(video_id)
            answer_text = "No answer available"
        else:
            answer_text = question_data.get(f'option {answer_idx}', 'Unknown option')

        # Store result
        result = {
            'video_id': video_id,
            'video_file': video_file,
            'question': question_data['question'],
            'options': {
                i: question_data.get(f'option {i}', '')
                for i in range(5)
            },
            'correct_answer_index': answer_idx,
            'correct_answer_text': answer_text,
            'google_drive_id': question_data.get('google_drive_id', '')
        }
        results.append(result)

    # Print summary
    print(f"\nProcessed {len(results)} videos successfully")
    if missing_questions:
        print(f"Missing questions for {len(missing_questions)} videos")
    if missing_answers:
        print(f"Missing answers for {len(missing_answers)} videos")

    return results, missing_questions, missing_answers


def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    VIDEOS_DIR = "videos"
    QUESTIONS_FILE = "questions.json"
    ANSWERS_FILE = "subset_answers.json"
    OUTPUT_FILE = "processed_videos.json"

    # Process videos
    results, missing_q, missing_a = process_videos(
        VIDEOS_DIR,
        QUESTIONS_FILE,
        ANSWERS_FILE
    )


    # Save all results
    save_results(results, OUTPUT_FILE)

    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total videos processed: {len(results)}")
    print(f"  Videos with questions: {len(results)}")
    print(f"  Videos with answers: {len([r for r in results if r['correct_answer_index'] is not None])}")

    if missing_q:
        print(f"\nVideos missing questions: {len(missing_q)}")
        print(f"  First few: {missing_q[:5]}")

    if missing_a:
        print(f"\nVideos missing answers: {len(missing_a)}")
        print(f"  First few: {missing_a[:5]}")
