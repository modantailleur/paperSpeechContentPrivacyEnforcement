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

input_script = "./dataset/metadata.csv"
asr_list = ["whisper", "fairseq", "w2v2", "crdnn"]

######### ORACLE #########
input_path = "./dataset/cityspeechmix/sonyc_librispeech_mixtures/"
output_dir = "./output/oracle/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### OURS #########
input_path = "./output/ours/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/ours/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### OURS ENVSS #########
input_path = "./output/ours_envss/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/ours_envss/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### COHEN #########
input_path = "./output/cohen/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/cohen/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### burkhardt #########
input_path = "./output/burkhardt/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/burkhardt/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

#####################################
####### "ABLATION" STUDY
######################################

######### OURS WITHOUT VAD #########
input_path = "./output/ours_novad/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/ours_novad/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### OURS WITHOUT SOURCE SEPARATION #########
input_path = "./output/ours_noss/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/ours_noss/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

# REMOVED FROM PAPER
# ######### OURS WITHOUT SMOOTHING #########
# input_path = "./output/ours_nosmooth/sonyc_librispeech_mixtures_anonymized/"
# output_dir = "./output/ours_nosmooth/"
# for asr in asr_list:
#     output_path = f"{output_dir}/wer_{asr}.csv"
#     command = [
#         "python", "compute_wer.py",
#         "-i", input_path,
#         "-is", input_script,
#         "-o", output_path,
#         "-asr", asr
#     ]
#     run_subprocess(command)

######### OURS WITH MIXING FRAMES #########
input_path = "./output/ours_with_mixframe/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/ours_with_mixframe/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

# #####################################################################################################################
# ##################################### DOUBLE REVERSE ###############################################################

######### OURS #########
input_path = "./output/rev_ours/sonyc_librispeech_mixtures_anonymized_anonymized/"
output_dir = "./output/rev_ours/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######### OURS WITH MIXING FRAMES #########
input_path = "./output/rev_ours_with_mixframe/sonyc_librispeech_mixtures_anonymized_anonymized/"
output_dir = "./output/rev_ours_with_mixframe/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)

######## NOISE #########
input_path = "./output/noise/sonyc_librispeech_mixtures_anonymized/"
output_dir = "./output/noise/"
for asr in asr_list:
    output_path = f"{output_dir}/wer_{asr}.csv"
    command = [
        "python", "compute_wer.py",
        "-i", input_path,
        "-is", input_script,
        "-o", output_path,
        "-asr", asr
    ]
    run_subprocess(command)