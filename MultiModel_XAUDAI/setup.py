import os
import sys
import subprocess

# Set the Hugging Face cache directory
os.environ['HF_HOME'] = 'cache/huggingface'

# Install required dependencies
print("Installing required dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0"])

# Now import and load the model
from transformers import AutoProcessor, AutoModelForImageTextToText

print("Downloading and loading Qwen2.5-VL model...")
vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# For CPU-only machines with limited resources
vl_model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="cpu",  # Use CPU explicitly
    low_cpu_mem_usage=True,  # Enable low memory usage mode
    offload_folder="offload_folder"  # Specify an offload folder for disk offloading
)
print("Model loaded successfully!")