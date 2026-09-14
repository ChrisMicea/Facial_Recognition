#!/usr/bin/env python3
"""
Facial Recognition Hub
Central interface for training, testing, and video face recognition.
Interactive mode - waits for user input commands.
"""

import sys


def print_usage():
    """Print usage information."""
    print("\n" + "="*50)
    print("Facial Recognition Hub - Interactive Mode")
    print("="*50)
    print("\nAvailable commands:")
    print("  train <train_dir>    - Train model with images from directory")
    print("  test <test_image>    - Test recognition on a static image")
    print("  video                - Run real-time video face recognition")
    print("  h/help               - Show this help message")
    print("  q/exit/quit          - Exit the program")
    print("\nExample:")
    print("  > train training_data/")
    print("  > test test_image.jpg")
    print("  > video")
    print("="*50 + "\n")


def main():
    print_usage()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split()
            command = parts[0].lower()
            
            if command in ['q', 'exit', 'quit']:
                print("Exiting...")
                break
            
            elif command in ['h', 'help']:
                print_usage()
            
            elif command == 'train':
                if len(parts) < 2:
                    print("Error: Please specify training directory")
                    print("Usage: train <train_dir>")
                    continue
                
                train_dir = parts[1]
                from src.train_encodings import train_and_save_encodings
                print(f"Training model with images from: {train_dir}")
                train_and_save_encodings(train_dir)
                print("Training complete! Encodings saved to encodings.bin and names.bin")
            
            elif command == 'test':
                if len(parts) < 2:
                    print("Error: Please specify test image")
                    print("Usage: test <test_image>")
                    continue
                
                test_image = parts[1]
                from src.recognize_static_image import recognize_faces_in_image
                print(f"Testing recognition on: {test_image}")
                recognize_faces_in_image(test_image)
            
            elif command == 'video':
                from src.recognize_on_video_feed import run_video_recognition
                print("Starting video face recognition...")
                print("Press 'ESC' to exit")
                run_video_recognition()
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'h' or 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
