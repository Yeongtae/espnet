"""
This code was developed with reference to https://github.com/Rayhane-mamah/Tacotron-2.
"""
from scipy.io.wavfile import write
import librosa
import numpy as np
import argparse
import os

sr = 16000
max_wav_value=32768.0
trim_fft_size = 1024
trim_hop_size = 256

# These are control parameters for trimming and skipping
trim_top_db = 23
skip_len = 14848

def clean_audios(baseDir, file_name):
    f = open(file_name,'r',encoding='utf-8')
    R = f.readlines()
    f.close()

    L = []
    for i, r in enumerate(R):
        wav_file = r.split('|')[0].replace('.\\','').replace('./','').replace('\\','/')
       # print(baseDir, wav_file)
        full_path = os.path.join(baseDir, wav_file)
        try:
            data, sampling_rate = librosa.core.load(full_path, sr)
            L.append('{}|{}'.format(wav_file, '|'.join(r.split('|')[1:]) ))
        except:
            pass
        if i % 100 ==0:
            print(i)
    tmp = file_name.split('.')
    tmp.insert(1,'_clean.')
    skipped_file_name = "".join(tmp)
    f = open(skipped_file_name,'w',encoding='utf-8')
    f.writelines(L)
    f.close()


if __name__ == "__main__":
    """
    usage
    python cleanMetadata.py -f=metadata.csv
    python cleanMetadata.py -b AIhub_NIKL_gcp_kakao_netmarble -f AIhub_NIKL_gcp_kakao_netmarble/metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseDir', type=str)
    parser.add_argument('-f', '--file', type=str,
                        help='Metadata file')
    args = parser.parse_args()
    file = args.file
    baseDir = args.baseDir

    clean_audios(baseDir, file)