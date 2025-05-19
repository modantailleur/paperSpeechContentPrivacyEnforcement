import subprocess
import os

pre_script = "CUDA_VISIBLE_DEVICES=1"

######################################################################################################################
###################################### SIMPLE REVERSE ###############################################################

# Path to the script to be launched
script_path = "./fig_anonymize-audio-for-figure.py"
input_path_mixed = "./example_audio/for_plot/"

# Helper function to build the command with pre_script
def run_subprocess(cmd_list):
    env = os.environ.copy()
    
    # Split pre_script into parts
    pre_parts = pre_script.split()
    cmd_prefix = []

    # Extract environment variable assignments
    for part in pre_parts:
        if '=' in part and not part.startswith('-'):
            key, val = part.split('=', 1)
            env[key] = val
        else:
            cmd_prefix.append(part)

    full_cmd = cmd_prefix + cmd_list
    print(full_cmd)
    subprocess.run(full_cmd, check=True, env=env)

###################
# Our method: vad, source separation, reverse transform with randwin and randhop
# Launch the script
output_path = f"./output/audio_for_plot/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")