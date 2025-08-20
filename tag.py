from huggingface_hub import HfApi
HfApi().create_tag(
    repo_id="shin1107/koch_new_fb_6d3",
    tag="v2.1",          # ← info.json の codebase_version の値に合わせる
    repo_type="dataset"
)
print("Tag created.")