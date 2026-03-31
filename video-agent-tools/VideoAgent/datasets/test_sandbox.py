#!/usr/bin/env python3
"""
Simple test script for SandboxManager.
Tests: create sandbox -> load video -> preprocess -> stop sandbox
"""

from sandbox_manager import SandboxManager
import os
import sys

def test_sandbox_manager():
    """Test basic sandbox manager functionality."""

    print("=" * 60)
    print("Testing SandboxManager")
    print("=" * 60)

    # Initialize SandboxManager
    print("\n1. Initializing SandboxManager...")
    manager = SandboxManager(
        base_dir='./test_sandboxes',
        show_tracking=False,
        tracking_fps=15,
        sample_num=5
    )
    print("   ✓ SandboxManager initialized")

    # Create a test sandbox
    sandbox_id = "test_sandbox_001"
    print(f"\n2. Creating sandbox '{sandbox_id}'...")
    sandbox_path = manager.create_sandbox(sandbox_id)
    print(f"   ✓ Sandbox created at: {sandbox_path}")
    print(f"   ✓ Sandbox exists: {manager.sandbox_exists(sandbox_id)}")

    # Load a video into the sandbox
    # You'll need to specify an actual video name from your videos directory
    video_name = input("\n3. Enter video name to test (e.g., 'video.mp4'): ").strip()

    if not video_name:
        print("   ⚠ No video name provided, skipping video load and preprocess")
        skip_preprocessing = True
    else:
        try:
            print(f"   Loading video '{video_name}'...")
            video_path = manager.load_video_into_sandbox(video_name, sandbox_id)
            print(f"   ✓ Video loaded to: {video_path}")
            print(f"   ✓ Video file exists: {os.path.exists(video_path)}")
            skip_preprocessing = False
        except FileNotFoundError as e:
            print(f"   ✗ Error: {e}")
            skip_preprocessing = True

    # Preprocess the video
    if not skip_preprocessing:
        print("\n4. Preprocessing video...")
        print("   This may take a while depending on video length...")
        try:
            manager.preprocess(sandbox_id)
            print("   ✓ Preprocessing completed")

            # Check if preprocessed files exist
            base_dir = os.path.join(manager.base_dir, sandbox_id)
            video_dir = os.path.join(base_dir, "video")
            if os.path.exists(video_dir):
                files = os.listdir(video_dir)
                print(f"   ✓ Preprocessed files ({len(files)}): {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
            else:
                print(f"   ⚠ Warning: Preprocessed directory not found at {video_dir}")
        except Exception as e:
            print(f"   ✗ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()

    manager.visual_question_answering(sandbox_id, ("what is the segment about?", 1))
    # Stop and clean up the sandbox
    print("\n5. Stopping sandbox...")
    choice = input("   Do you want to delete the test sandbox? (y/n): ").strip().lower()

    if choice == 'y':
        try:
            manager.stop_sandbox(sandbox_id)
            print(f"   ✓ Sandbox '{sandbox_id}' stopped and removed")
            print(f"   ✓ Sandbox exists: {manager.sandbox_exists(sandbox_id)}")
        except Exception as e:
            print(f"   ✗ Error stopping sandbox: {e}")
    else:
        print(f"   ℹ Sandbox kept at: {sandbox_path}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_sandbox_manager()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
