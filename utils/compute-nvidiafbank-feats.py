#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy
import torch

from espnet.transform.layers import TacotronSTFT
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper


def get_parser():
    parser = argparse.ArgumentParser(
        description='compute FBANK feature from WAV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int,
                        help='Sampling frequency')
    parser.add_argument('--fmax', type=int, default=None, nargs='?',
                        help='Maximum frequency')
    parser.add_argument('--fmin', type=int, default=None, nargs='?',
                        help='Minimum frequency')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='Number of mel basis')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=None, nargs='?',
                        help='Analisys window length in point')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for output. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or '
                             'gzip-level(if hdf5)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--normalize', choices=[1, 16, 24, 32], type=int,
                        default=None,
                        help='Give the bit depth of the PCM, '
                             'then normalizes data to scale in [-1,1]')
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
    parser.add_argument(
        '--segments', type=str,
        help='segments-file format: each line is either'
             '<segment-id> <recording-id> <start-time> <end-time>'
             'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    return parser

class hparams():
    def __init__(self, n_mel_channels, sampling_rate, filter_length, hop_length, win_length, max_abs_mel_value, mel_fmin, mel_fmax):
        super(hparams)
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_abs_mel_value = max_abs_mel_value
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

def main():
    parser = get_parser()
    args = parser.parse_args()
    stft_config = hparams(args.n_mels, args.fs, args.n_fft, args.n_shift, args.win_length, 4.0, args.fmin, args.fmax)
    fExtractor =TacotronSTFT(stft_config).cpu()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with kaldiio.ReadHelper(args.rspecifier,
                            segments=args.segments) as reader, \
            file_writer_helper(args.wspecifier,
                               filetype=args.filetype,
                               write_num_frames=args.write_num_frames,
                               compress=args.compress,
                               compression_method=args.compression_method
                               ) as writer:
        for utt_id, (rate, array) in reader:
            assert rate == args.fs
            array = array.astype(numpy.float32)
            if args.normalize is not None and args.normalize != 1:
                array = array / (1 << (args.normalize - 1))

            #print('wave: ', array.max(), array.mean(), array.min(), array.shape)

            wave = torch.from_numpy(array).unsqueeze(0)
            lmspc2 = fExtractor.mel_spectrogram(y=wave, isDebugging=False).squeeze_(0).transpose_(0,1)
            lmspc2 = lmspc2.numpy()
            writer[utt_id] = lmspc2
            #print('lmspc2: ', lmspc2.max(), lmspc2.mean(), lmspc2.min(), lmspc2.shape)


if __name__ == "__main__":
    main()
