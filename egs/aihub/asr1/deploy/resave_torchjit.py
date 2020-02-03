import os
import sys
import argparse
import torch
import numpy as np
import logging
import json
import configargparse
import importlib

import librosa
from e2e import Encoder, DecoderRNNT, E2E
import e2e
from io_utils import LoadInputsAndTargets

def get_model_conf(conf_path=None):

    with open(conf_path, "rb") as f:
        logging.warning("reading a config file from " + conf_path)
        confs = json.load(f)
    idim, odim, args = confs
    return idim, odim, argparse.Namespace(**args)

def load_pytorch_resave(model_path, conf_path):
    idim, odim, train_args = get_model_conf(os.path.abspath(conf_path))

    logging.info('reading model parameters form ' + conf_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
        logging.waring('model module not exists')

    subsample = train_args.subsample.split('_')
    susample = list(map(int, subsample))
    encoder = Encoder(train_args.etype, idim, train_args.elayers, train_args.eunits, train_args.eprojs,
                        subsample, train_args.dropout_rate, vgg_channel=train_args.vgg_ichannels)
    blank=0

    # eprojs, odim, dtype, dlayers, dunits, blank, embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0, rnnt_type='warp-transducer'
    decoder = DecoderRNNT(train_args.eprojs, odim, train_args.dtype, train_args.dlayers, train_args.dunits, blank,
                              train_args.dec_embed_dim, train_args.joint_dim, train_args.dropout_rate_decoder,
                              train_args.dropout_rate_embed_decoder, train_args.rnnt_type)

    model = E2E(idim, odim, train_args, encoder, decoder)
    model.load_state_dict(torch.load(os.path.abspath(model_path), map_location=torch.device('cpu')))

    checkpoint = {
            'enc': model.enc.state_dict(),
            'dec': model.dec.state_dict()
            }

    return checkpoint, idim, odim, train_args

def load_pytorch_model(load_file, idim, odim, args):

    checkpoint = torch.load(load_file, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['enc']
    decoder_sd = checkpoint['dec']

    #print("decoder_sd={}".format(decoder_sd))
    subsample = args.subsample.split('_')
    susample = list(map(int, subsample))
    encoder = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample,
                        args.dropout_rate, vgg_channel=args.vgg_ichannels)
    blank=0

    # eprojs, odim, dtype, dlayers, dunits, blank, embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0, rnnt_type='warp-transducer'
    decoder = DecoderRNNT(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, blank,
                              args.dec_embed_dim, args.joint_dim, args.dropout_rate_decoder,
                              args.dropout_rate_embed_decoder, args.rnnt_type)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    encoder.eval()
    decoder.eval()
    e2e = E2E(idim, odim, args, encoder, decoder)
    e2e.eval()
    #print(e2e)
    return e2e

def load_torchscript_model(load_file, idim, odim, args, output):

    checkpoint = load_file
    encoder_sd = checkpoint['enc']
    decoder_sd = checkpoint['dec']

    subsample = args.subsample.split('_')
    susample = list(map(int, subsample))
    encoder = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample,
                        args.dropout_rate, vgg_channel=args.vgg_ichannels)
    blank=0
    # eprojs, odim, dtype, dlayers, dunits, blank, embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0, rnnt_type='warp-transducer'
    decoder = DecoderRNNT(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, blank,
                              args.dec_embed_dim, args.joint_dim, args.dropout_rate_decoder,
                              args.dropout_rate_embed_decoder, args.rnnt_type)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    print(encoder)
    print(decoder)
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    encoder.eval()
    decoder.eval()

    traced_encoder = torch.jit.script(encoder)  # traced_encoder (12,76,320)
    #en_output, en_length = traced_encoder(ex_input, length)
    traced_decoder = torch.jit.script(decoder)  # traced_encoder (12,76,320)

    #print("encoder.....................={}".format(traced_encoder))
    #print("decoder......................={}".format(traced_decoder))

    # E2E build
    #idim, odim, train_args = get_model_conf(os.getcwd() + '/exp/train-clean_pytorch_train_transducer_specaug/results/model.json')
    e2e = E2E(idim, odim, args, traced_encoder, traced_decoder)
    # e2e.eval()
    traced_e2e = torch.jit.script(e2e)
    traced_e2e.to('cpu')
    traced_e2e.eval()
    print('scripted graph:\n', traced_e2e)
    traced_e2e.save(os.getcwd()+'/'+output)
    return traced_e2e


def save_torchscript(args):
    """Decode with the given args.
    Args:
        args (namespace): The program arguments.
    """

    # Load original pytorch model.
    checkpoint, idim, odim, train_args = load_pytorch_resave(args.model_torch, args.model_json)

    # Get torch traced/scripted model. And save torch script model.
    model = load_torchscript_model(checkpoint, idim, odim, train_args, args.output)
    model.eval()

def get_parser():
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # task related
    parser.add_argument('--model-json', type=str,
                        help='Filename of model data (json)')
    parser.add_argument('--model-torch', type=str,
                        help='Filename of pytorch model (encdec.pt)')
    parser.add_argument('--output', type=str,
                        help='Filename of torchscript model (torchscript.pt)')
    return parser

def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    save_torchscript(args)

## python resave_torchjit.py --model-json model.json --model-torch model.loss.best --output test.pt
if __name__ == '__main__':
    main(sys.argv[1:])


