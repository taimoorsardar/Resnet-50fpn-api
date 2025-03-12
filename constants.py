import os
# Supabase credentials
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"

# Define the target categories, path to the checkpoint, and model module.
CATS = ["suspicious_site"]
STATE_DICT_PATH = os.path.join("weights", "checkpoint.pth")
MODEL_MODULE = "architecture.resnet50_fpn"