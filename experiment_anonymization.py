import subprocess
import os

pre_script = "CUDA_VISIBLE_DEVICES=1"

######################################################################################################################
###################################### SIMPLE REVERSE ###############################################################

# Path to the script to be launched
script_path = "./anonymize_audio_folder.py"
input_path_mixed = "./dataset/cityspeechmix/sonyc_librispeech_mixtures/"
input_path_untouched = "./dataset/cityspeechmix/sonyc_unmixed_subset/"

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

# #######################################
# ######### MAIN EXPERIMENT
# ########################################


###################
# Our method: vad, source separation, reverse transform
# Launch the script
output_path = f"./output/ours/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

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

output_path = f"./output/ours/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")

###################
# Our method with env source sep: vad, env source separation, reverse transform
# Launch the script
output_path = f"./output/ours_envss/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "--sep_method", "env",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/ours_envss/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "--sep_method", "env",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")

###################
# burkhardt method: no vad, no source separation, reverse transform with fixed win, no hop
# paper: MASKING SPEECH CONTENTS BY RANDOM SPLICING: IS EMOTIONAL EXPRESSION PRESERVED?
# Launch the script
output_path = f"./output/burkhardt/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-vad", "False",
        "-sep", "False",
        '-reverse', 'forward',
        '-segmin', "10.0",
        '-segmax', "10.0",
        '-seghopmin', "1.0",
        '-seghopmax', "1.0",
        '-framemin', "0.0",
        '-framehopmin', "1.0",
        '-mixframe', "True",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/burkhardt/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "-vad", "False",
        "-sep", "False",
        '-reverse', 'forward',
        '-segmin', "10.0",
        '-segmax', "10.0",
        '-seghopmin', "1.0",
        '-seghopmax', "1.0",
        '-framemin', "0.0",
        '-framehopmin', "1.0",
        '-mixframe', "True",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")

###################
# cohen method: no vad, source separation, mfcc transform
# Launch the script
output_path = f"./output/cohen/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-vad", "False",
        "-method", "mfcc",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/cohen/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "-vad", "False",
        "-method", "mfcc",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")


# #####################################
# ####### "ABLATION" STUDY
# ######################################

# REMOVED FROM PAPER
# ###################
# # Our method without smoothening
# output_path = f"./output/ours_nosmooth/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# # Ensure the output directory exists
# os.makedirs(output_path, exist_ok=True)
# try:
#     run_subprocess([
#         "python3", script_path,
#         "-i", input_path_mixed,
#         "-o", output_path,
#         '-seghopmin', "1.0",
#         '-seghopmax', "1.0",
#         '-framemin', "0.0",
#         '-framehopmin', "1.0",
#     ])
#     print("Script executed successfully for mixed input.")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred while executing the script for mixed input: {e}")

# output_path = f"./output/ours_nosmooth/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# # Ensure the output directory exists
# os.makedirs(output_path, exist_ok=True)
# try:
#     run_subprocess([
#         "python3", script_path,
#         "-i", input_path_untouched,
#         "-o", output_path,
#         '-seghopmin', "1.0",
#         '-seghopmax', "1.0",
#         '-framemin', "0.0",
#         '-framehopmin', "1.0",
#     ])
#     print("Script executed successfully for untouched input.")
# except subprocess.CalledProcessError as e:
#     print(f"An error occurred while executing the script for untouched input: {e}")


###################
# Our method without VAD
output_path = f"./output/ours_novad/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-vad", "False",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/ours_novad/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "-vad", "False",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")

###################
# Our method without source separation
output_path = f"./output/ours_noss/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-sep", "False",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/ours_noss/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "-sep", "False",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")

#####################################
####### "ROBUSTNESS" STUDY
######################################

###################
# Our method with mixing frames
output_path = f"./output/ours_with_mixframe/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-mixframe", "True",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")

output_path = f"./output/ours_with_mixframe/{os.path.basename(os.path.normpath(input_path_untouched))}_anonymized"
# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_untouched,
        "-o", output_path,
        "-mixframe", "True",
    ])
    print("Script executed successfully for untouched input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for untouched input: {e}")


# #####################################################################################################################
# ##################################### DOUBLE REVERSE ###############################################################

###################
# Our method: vad, source separation, reverse transform with randwin and randhop
# Launch the script
input_path_mixed = "./output/ours/sonyc_librispeech_mixtures_anonymized/"
output_path = f"./output/rev_ours/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

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

###################
# Our method without mixing frames
input_path_mixed = "./output/ours_with_mixframe/sonyc_librispeech_mixtures_anonymized/"
output_path = f"./output/rev_ours_with_mixframe/{os.path.basename(os.path.normpath(input_path_mixed))}_anonymized/"

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
try:
    run_subprocess([
        "python3", script_path,
        "-i", input_path_mixed,
        "-o", output_path,
        "-mixframe", "True",
    ])
    print("Script executed successfully for mixed input.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the script for mixed input: {e}")