import os
import sys

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')

if sys.version_info[1] >= 10: # >=3.10 for genesis and isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        SIMULATOR = "genesis"
    elif simulator_type == "isaacsim":
        SIMULATOR = "isaacsim"
    else:
        raise ValueError("Unsupported SIMULATOR type. Please set the SIMULATOR environment variable to 'genesis' or 'isaacsim'.")
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"

if SIMULATOR == "genesis":
    import genesis as gs
elif SIMULATOR == "isaacgym":
    import isaacgym