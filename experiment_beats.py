import subprocess
import os

pre_script = "CUDA_VISIBLE_DEVICES=1"

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

#####################################
####### MAIN EXPERIMENT
######################################

######### ORACLE #########
input_path = "./dataset/"
output_path = "./output/oracle/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

######## OURS #########
input_path = "./output/ours/"
output_path = "./output/ours/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

# ######### OURS ENVSS #########
input_path = "./output/ours_envss/"
output_path = "./output/ours_envss/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

######### COHEN #########
input_path = "./output/cohen/"
output_path = "./output/cohen/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

######### burkhardt #########
input_path = "./output/burkhardt/"
output_path = "./output/burkhardt/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)


#######################################
######### "ABLATION" STUDY
########################################

######### OURS WITHOUT VAD #########
input_path = "./output/ours_novad/"
output_path = "./output/ours_novad/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

# ######### OURS WITHOUT SOURCE SEP #########
input_path = "./output/ours_noss/"
output_path = "./output/ours_noss/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

# REMOVED FROM PAPER
# ######### OURS WITHOUT SMOOTHING #########
# input_path = "./output/ours_nosmooth/"
# output_path = "./output/ours_nosmooth/logits.csv"
# command = [
#     "python", "compute_beats.py",
#     "-i", input_path,
#     "-o", output_path
# ]

# run_subprocess(command)

######### OURS WITH MIXING WINDOWS #########
input_path = "./output/ours_with_mixframe/"
output_path = "./output/ours_with_mixframe/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)

######### NOISE #########
input_path = "./output/noise/"
output_path = "./output/noise/logits.csv"
command = [
    "python", "compute_beats.py",
    "-i", input_path,
    "-o", output_path
]

run_subprocess(command)