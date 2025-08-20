HF hub でフォルダを作る


# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Login with your Hugging Face credentials
hf auth login

# Push your dataset files
hf upload shin1107/koch_new_fb_6d2 . --repo-type=dataset