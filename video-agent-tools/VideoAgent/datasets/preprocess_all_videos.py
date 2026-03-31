#!/usr/bin/env python3
"""
Script to preprocess all videos in the EgoSchema directory sequentially.
"""

import os
from pathlib import Path
from sandbox_manager import SandboxManager

def main():
    # Directory containing the videos
    video_dir = "/path/to/tvcache/train/EgoSchema/videos"

    # Initialize sandbox manager
    sandbox_manager = SandboxManager(
        base_dir='./sandboxes',
        show_tracking=False,
        tracking_fps=15,
        sample_num=5
    )

    # Get all video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

    print(f"Found {len(video_files)} videos to process")

    # Process each video sequentially
    for idx, video_file in enumerate(video_files, 1):
        # Use video filename (without extension) as sandbox_id
        sandbox_id = Path(video_file).stem

        print(f"\n{'='*80}")
        print(f"Processing video {idx}/{len(video_files)}: {video_file}")
        print(f"Sandbox ID: {sandbox_id}")
        print(f"{'='*80}")

        try:
            # Create sandbox
            print(f"Creating sandbox for {sandbox_id}...")
            sandbox_manager.create_sandbox(sandbox_id)

            # Load video into sandbox
            print(f"Loading video {video_file} into sandbox...")
            sandbox_manager.load_video_into_sandbox(video_file, sandbox_id)

            # Preprocess the video
            print(f"Preprocessing {video_file}...")
            sandbox_manager.preprocess(sandbox_id)

            print(f"✓ Successfully preprocessed {video_file}")

        except Exception as e:
            print(f"✗ Error processing {video_file}: {str(e)}")
            continue

    print(f"\n{'='*80}")
    print(f"Finished processing all videos!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
