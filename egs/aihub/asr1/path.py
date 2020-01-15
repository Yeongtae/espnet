#!/bin/bash
import os
import subprocess

def set_path():
    print('set kaldi root path')

    MAIN_ROOT = os.getcwd() + '/../../..'
    KALDI_ROOT = MAIN_ROOT + '/tools/kaldi'

    os.environ['KALDI_ROOT'] = MAIN_ROOT+'/tools/kaldi'

    if not os.environ.get("KALDI_ROOT") is None:
        print('kaldi root='+KALDI_ROOT)
        os.environ['PATH'] += os.pathsep + os.getcwd() + '/utils/'
        os.environ['PATH'] += os.pathsep + os.getcwd() + '/steps/'
        os.environ['PATH'] += os.pathsep + KALDI_ROOT + '/tools/openfst/bin'
        os.environ['PATH'] += os.pathsep + KALDI_ROOT + '/tools/sctk/bin'

    temp_path = KALDI_ROOT+'/tools/config/common_path.sh'
    if not os.path.exists(temp_path):
        print('The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit()')
        exit()

    os.environ['LC_ALL'] = 'C'

    print(os.environ.get('LC_ALL'))
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + MAIN_ROOT + '/tools/chainer_ctc/ext/warp-ctc/build'
    #if os.path.exists(MAIN_ROOT + '/tools/venv/etc/profile.d/conda.sh'):
    #    import shlex
    #    subprocess.Popen('source '+MAIN_ROOT+'/tools/venv/etc/profile.d/conda.sh', shell=True)
        #os.system('source '+MAIN_ROOT+'/tools/venv/etc/profile.d/conda.sh')
        #subprocess.Popen('conda '+'deactivate', shell=True)
        #subprocess.Popen('conda '+'activate', shell=True)

    #subprocess.Popen(['/bin/bash','source ../../../tools/venv/bin/activate'], shell=True)

    os.environ['PATH'] += os.pathsep + MAIN_ROOT + '/utils'
    os.environ['PATH'] += os.pathsep + MAIN_ROOT + '/espnet/bin'
    os.environ['PATH'] += os.pathsep + '/home/hwak1234/espnet/'
    os.environ['PATH'] +=  os.pathsep + MAIN_ROOT + '/tools/chainer_ctc/ext/warp-transducer/pytorch_binding'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['PYTHONIOENCODING'] = 'UTF-8'
    os.environ['PYTHONPATH'] = os.pathsep + MAIN_ROOT
    #os.environ['PYTHONPATH'] += os.pathsep + MAIN_ROOT + '/tools/chainer_ctc/ext/warp-transducer/pytorch_binding'
    print(os.environ.get('OMP_NUM_THREADS'))
    print('confirm='+os.environ.get('PATH'))

    #subprocess.Popen(['/bin/bash ./path.sh'], shell=True)
    #os.system("sh "+os.getcwd()+'/path.sh')
