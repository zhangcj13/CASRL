
import sys, warnings
# if sys.version_info < (3,0): warnings.warn("Optimized for Python3. Performance may suffer under Python2.", Warning)
import os
os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase1_GA3C'
# os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase2_GA3C'

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from gym_collision_avoidance.envs import Config
import torch


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from ga3c.Server import Server

if __name__ == '__main__':
    Server().main()

    print("killing process...")
    import os, signal
    os.kill(os.getpid(),signal.SIGKILL)

    print("Train done!")

    # "tensorboard --logdir= , http://localhost:6006"

