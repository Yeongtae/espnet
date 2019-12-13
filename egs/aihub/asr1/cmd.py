import os

def set_cmd():

    # Select the backend used by run.sh from "local", "sge", "slurm", or "ssh"
    cmd_backend='local'

    if cmd_backend == 'local':
        print('local set path')
        os.environ['train_cmd'] = "run.pl"
        os.environ['cuda_cmd'] = "run.pl"
        os.environ['decode_cmd'] = "run.pl"

    elif cmd_backend == 'slurm':
        os.environ['train_cmd'] = "ssh.pl"
        os.environ['cuda_cmd'] = "ssh.pl"
        os.environ['decode_cmd'] = "ssh.pl"
    elif cmd_backend == 'jhu':
        os.environ['train_cmd'] = "queue.pl --mem 2G"
        os.environ['cuda_cmd'] = "queue-freegpu.pl --mem 2G --gpu 1 --config conf/gpu.conf"
        os.environ['decode_cmd'] = "queue.pl --mem 4G"
    else:
        print("{}: Error: Unknown cmd_backend={} ".format(0, cmd_backend) )
        return 1
