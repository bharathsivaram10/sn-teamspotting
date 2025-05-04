import gradio as gr
import torch
from PIL import Image
import cv2
import numpy as np
import time
import os
import json
import tempfile
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """Load the model and processor from HuggingFace."""
    print(f"Loading model {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(DEVICE)
    print("Model loaded successfully.")
    
    return model, processor

def generate_response(video_path):
    """Process a video file to detect soccer actions."""
    model, processor = load_model()
    
    # Extract frames and timestamps
    try:
        frames, timestamps = extract_frames_times(video_path)
    except Exception as e:
        return f"Error extracting frames: {str(e)}"
    
    if not frames:
        return "No frames were extracted from the video."
    
    # Create data structure for annotations
    annotations = []
    
    print(f"Processing {len(frames)} frames...")
    
    # Process each frame
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Processing frame {i+1}/{len(frames)}")
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        system_prompt = '''You are a soccer video assistant, and your job is to identify key soccer actions. Here are the definitions for each action we are interested in:
        PASS: A player kicks the ball to a teammate to maintain possession
        DRIVE: An attacking dribble taken
        HEADER: Striking the ball using the head, usually to pass, clear, or score
        HIGH PASS: A pass lofted through the air to reach a teammate over distance or defenders
        OUT: The ball goes completely over the touchline or goal line, stopping play
        CROSS: A pass from the side of the field into the opponent's penalty area
        THROW IN: A two-handed overhead throw used to return the ball into play after it goes out on the sideline
        SHOT: An attempt to score by kicking or heading the ball toward the goal
        BALL PLAYER BLOCK: A player obstructs the ball or ball carrier to prevent progress
        PLAYER SUCCESSFUL TACKLE: A player legally takes the ball away from an opponent
        FREE KICK: A kick awarded after a foul, allowing an unopposed restart
        GOAL: When the entire ball crosses the goal line between the posts and under the crossbar
        NONE: None of the above actions'''
        
        user_prompt = "Identify whether there was an action taken, and if so, what team (left or right). Return in the format 'ACTION-team'. If no action is taken return 'NONE'"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        input_data = processor(text=prompt, images=pil_image, return_tensors="pt")
        input_data = {k: v.to(DEVICE) for k, v in input_data.items()}
        
        # Generate output
        with torch.no_grad():
            output = model.generate(
                **input_data,
                max_new_tokens=32,
                do_sample=False
            )
            
        result = processor.decode(output[0], skip_special_tokens=True)
        result = result.strip()
        
        # Parse result
        action = "NONE"
        team = "N/A"
        
        if result != "NONE" and "-" in result:
            action, team = result.split("-", 1)
        elif result != "NONE":
            action = result
        
        # Skip if no action detected
        if action == "NONE":
            continue
            
        # Format timestamp as MM:SS
        minutes = int(timestamp) // 60
        seconds = int(timestamp) % 60
        game_time = f"{minutes:02d}:{seconds:02d}"
        
        # Position in milliseconds
        position_ms = int(timestamp * 1000)
        
        # Add to annotations
        annotation = {
            "gameTime": game_time,
            "label": action,
            "position": str(position_ms),
            "team": team.lower() if team != "N/A" else ""
        }
        
        annotations.append(annotation)
    
    # Create a JSON structure
    result_json = {"annotations": annotations}
    
    # Create a temporary file for output
    import json
    output_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
    output_path = output_file.name
    
    # Write JSON to file
    json.dump(result_json, output_file, indent=4)
    output_file.close()
    
    print(f"Processing complete. Results saved to {output_path}")
    
    return output_path

def extract_frames_times(video_path):
    """
    Extract frames at 10 fps and return a list of frames and corresponding timestamps.
    If video length over 5 minutes, return an error.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Check if video is too long (over 5 minutes)
    if duration > 300:  # 5 minutes = 300 seconds
        cap.release()
        raise Exception("Error: Video is too long (over 5 minutes). Please upload a shorter video.")
    
    # Calculate frame extraction rate (10 fps)
    target_fps = 10
    frame_interval = max(1, int(fps / target_fps))
    
    frames = []
    timestamps = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frames.append(frame)
            timestamps.append(frame_idx / fps)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames from video (original fps: {fps}, target fps: {target_fps})")
    
    return frames, timestamps

# Create Gradio interface allowing user to upload a video
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Video(label="Upload Soccer Video (max 5 minutes)"),
    outputs=gr.File(label="Action Detection Results (JSON)"),
    title="Soccer Action Detection",
    description="Upload a soccer video to detect actions such as passes, shots, headers, etc.",
    examples=[],
    cache_examples=False,
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)