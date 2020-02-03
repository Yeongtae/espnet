import os
import argparse
import torch
import numpy as np
import importlib
import subprocess
import sys
import configargparse
import logging
import json
from path import set_path

from e2e import Encoder, DecoderRNNT, E2E
import e2e
from io_utils import LoadInputsAndTargets

sys.path.append('../../../utils')
def get_model_conf(conf_path=None):

    with open(conf_path, "rb") as f:
        logging.warning("reading a config file from " + conf_path)
        confs = json.load(f)
    idim, odim, args = confs
    return idim, odim, argparse.Namespace(**args)

def load_pytorch_model(load_file, idim, odim, args):

    checkpoint = torch.load(load_file, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['enc']
    decoder_sd = checkpoint['dec']

    #print("decoder_sd={}".format(decoder_sd))
    subsample = args.subsample.split('_')
    susample = list(map(int, subsample))
    encoder = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
    blank=0

    # eprojs, odim, dtype, dlayers, dunits, blank, embed_dim, joint_dim, dropout=0.0, dropout_embed=0.0, rnnt_type='warp-transducer'
    decoder = DecoderRNNT(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, blank,
                              args.dec_embed_dim, args.joint_dim, args.dropout_rate_decoder, args.dropout_rate_embed_decoder, args.rnnt_type)

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

def to_torch_tensor(x):
    """Change to torch.Tensor or ComplexTensor from numpy.ndarray.

    Args:
        x: Inputs. It should be one of numpy.ndarray, Tensor, ComplexTensor, and dict.

    Returns:
        Tensor or ComplexTensor: Type converted inputs.

    Examples:
        >>> xs = np.ones(3, dtype=np.float32)
        >>> xs = to_torch_tensor(xs)
        tensor([1., 1., 1.])
        >>> xs = torch.ones(3, 4, 5)
        >>> assert to_torch_tensor(xs) is xs
        >>> xs = {'real': xs, 'imag': xs}
        >>> to_torch_tensor(xs)
        ComplexTensor(
        Real:
        tensor([1., 1., 1.])
        Imag;
        tensor([1., 1., 1.])
        )

    """
    # If numpy, change to torch tensor
    if isinstance(x, np.ndarray):
        if x.dtype.kind == 'c':
            # Dynamically importing because torch_complex requires python3
            from torch_complex.tensor import ComplexTensor
            return ComplexTensor(x)
        else:
            return torch.from_numpy(x)

    # If {'real': ..., 'imag': ...}, convert to ComplexTensor
    elif isinstance(x, dict):
        # Dynamically importing because torch_complex requires python3
        from torch_complex.tensor import ComplexTensor

        if 'real' not in x or 'imag' not in x:
            raise ValueError("has 'real' and 'imag' keys: {}".format(list(x)))
        # Relative importing because of using python3 syntax
        return ComplexTensor(x['real'], x['imag'])

    # If torch.Tensor, as it is
    elif isinstance(x, torch.Tensor):
        return x
    else:
        return x

def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    print("hyp process={}".format(hyp))
    score = hyp['score']

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score

def add_results_to_json(js, nbest_hyps, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]): List of hypothesis for multi_speakers: nutts x nspkrs.
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js['output']) > 0:
            out_dic = dict(js['output'][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {'name': ''}

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        print('ground truth sentence={}'.format(out_dic['text']))
        print('output sentence={}'.format(out_dic['rec_text']))
        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js

def evaluate_post_train(args, dump):
    """Evaluate with the given args.

    Args:
        args (namespace): The program arguments.
    """

    idim, odim, train_args = get_model_conf(args.model_json)

    # Get model
    model = load_pytorch_model(args.model_pytorch, idim, odim, train_args)
    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)

    torch.quantization.prepare(model, inplace=True)

    # Calibrate first


def evaluate(args, dump):
    """Evaluate with the given args.

    Args:
        args (namespace): The program arguments.
    """

    idim, odim, train_args = get_model_conf(args.model_json)

    # Get model
    model = load_pytorch_model(args.model_pytorch, idim, odim, train_args)
    quan_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)

    #quan_model = model
    quan_model.eval()

    # read json test, eval data
    with open(dump, 'rb') as f:
        js = json.load(f)['utts']

    new_js={}
    load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=False, sort_in_input_length=False,
            preprocess_conf=None if args.preprocess_conf is None else args.preprocess_conf,
            preprocess_args={'train': False})

    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)
            feat = feat[0][0]
            #print('feat={}'.format(feat))
            subsample = np.ones(train_args.elayers+1, dtype=np.int)
            ilens = [feat.shape[0]]
            ilens = torch.tensor(ilens, dtype=torch.int32)
            feat = feat[::subsample[0], :]
            h = to_torch_tensor(feat).float().to('cpu')
            yseq = quan_model(h, ilens)
            score = 0
            nbest_hyps = [{'score':score, 'yseq': yseq.tolist()}]
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)
            #print('new_js={}'.format(new_js))

            ##if not os.path.exists(args.result_label)
    with open(args.result_label, 'ab') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))

def get_parser():
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--recog-json', type=str,
            help='Filename of model recog data (json)')
    parser.add_argument('--model-json', type=str,
            help='Filename of model data (json)')
    parser.add_argument('--result-label', type=str,
            help='Filename of result label data (json)')
    parser.add_argument('--preprocess-conf', type=str,
            help='Filename of preprocess conf (conf)')
    parser.add_argument('--model-pytorch', type=str,
            help='Filename of pytorch model (pt)')

    return parser

def main(args):
    set_path()
    parser = get_parser()
    args = parser.parse_args(args)
    preprocess_config='conf/specaug.yaml'
    traindata='train-clean'
    backend='pytorch'

    if preprocess_config=="":
        expname = traindata+'_'+backend+'_'+'train_transducer'
    else:
        preconfig = preprocess_config.split('/')[-1].split('.')[0]
        expname = traindata+'_'+backend+'_'+'train_transducer_'+preconfig

    main_expdir = 'exp/'+expname

    dict_path = 'data/lang_1char/'+'train-clean_units.txt'
    recog_dataset = ['train-test', 'train-val']
    decode_config='conf/tuning/transducer/decode_transducer.yaml'

    for rtask in recog_dataset:
        dump = 'dump/'+rtask+'/deltafalse/data.json'
        #evaluate(args, dump)
        evaluate_post_train(args, dump)

        #decode_config_dir=decode_config.split('/')[-1].split('.')[0]
        #decode_dir='decode_'+rtask+'_'+decode_config_dir

        #score_cmd = 'score_sclite.sh '+'--wer false '+main_expdir+'/'+decode_dir+' '+dict_path
        #subprocess.call([score_cmd], shell=True)

    print('finished')

if __name__ == '__main__':
    main(sys.argv[1:])
