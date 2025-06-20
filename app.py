import os
import argparse
import torch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hausa Health Assistant Training App")
    parser.add_argument("--share", action="store_true", default=True, help="Create a shareable link")
    
    args = parser.parse_args()
    
    # Clear any CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set environment variables to improve stability
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(f"Starting Hausa Health Assistant app...")
    
    # Import here to avoid loading the model before clearing CUDA cache
    from gradio_app import app
    
    # Launch the app
    app.launch(share=args.share)

if __name__ == "__main__":
    main()