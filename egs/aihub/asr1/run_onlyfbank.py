import os
import sys
import subprocess
from path import set_path
from cmd import set_cmd
from local import prepare_dataset_for_kaldiio_

#from espnet.utils.cli_utils import strtobool
sys.path.append('../../../utils')  ## espnet/utils included

def main():
    ngpu='1'
    stage=0

    train_config='conf/tuning/transducer/train_transducer.yaml'
    decode_config='conf/tuning/transducer/decode_transducer.yaml'
    #preprocess_config='conf/specaug.yaml'
    preprocess_config=""

    backend='pytorch'
    debugmode='1'
    verbose='0'
    resume=''

    expname = 'nvidia_STFT_aihub_org'
    #expname = 'train_pytorch_train_transducer'

    traindata='train'
    traindt='validation'
    do_delta='false'

    report_cer = 'true'
    report_wer = 'true'

    dump_dir_tr = 'dump/'+traindata+'/delta'+do_delta
    dump_dir_dt = 'dump/'+traindt+'/delta'+do_delta

    stage1_dataset=['train','test','validation']
    basis_dataset = ['train']
    stage2_dataset=['train','test','validation']
    recog_dataset=['test']

    set_path()
    set_cmd()

    print(os.environ.get('PATH'))

# stage 0: Data preparation
    if stage <= 0:
        print('stage 0: Data preparation')
        currpath = os.getcwd() + '/data'
        prepare_dataset_for_kaldiio_.run('/audio/projects/rnn-transducer-old/AIhub', '/audio/projects/rnn-transducer-old/AIhub/metadata.csv')
    else:
        print('skip! stage 0: Data Preparation')

## stage 1: Feature Generation
    if stage <= 1:
        print('stage 1: Feature Generation')

        fbankdir='/audio/data/aihub/fbank'
        if not os.path.exists(fbankdir):
            os.makedirs(fbankdir)

        exp_path = os.getcwd() + '/exp/make_fbank/'

        for x in stage1_dataset:

            if not os.path.exists(exp_path+x):
                os.makedirs(exp_path+x)

            ## Execute make_fbank_pitch.sh
            curr = os.getcwd()
            cmd_o = ' '+'--cmd'+' '+os.environ.get('train_cmd')
            cmd_nj = ' '+'--nj'+' '+'32'
            cmd_other = ' '+' '
            data_src = ' '+'data/'+x
            log = ' '+'exp/make_fbank/'+x
            # ret = subprocess.call([curr+'/../../../utils/make_fbank.sh'+ \
            #         cmd_o+cmd_nj+cmd_other+data_src+log+' '+fbankdir], shell=True)
            ret = subprocess.call([curr+'/../../../utils/make_nvidiafbank.sh'+ \
                    cmd_o+cmd_nj+cmd_other+data_src+log+' '+fbankdir], shell=True)
            # ret = subprocess.call([curr + '/../../../utils/make_nvidiafbank.sh' + \
            #                        data_src ], shell=True)

            ## Execute fix_data_dir.sh

            ret = subprocess.call([curr+'/utils/fix_data_dir_myown.sh'+data_src], shell=True)

            ## Compute global CMVN(only train-clean)

            if x in basis_dataset:
                ret = subprocess.call(['compute-cmvn-stats'+' '+'scp:data/'+x+'/feats.scp'+data_src+'/cmvn.ark'], shell=True)

            ## dump features for training
            do_delta = 'false'
            dump_dir = 'dump/'+x+'/delta'+do_delta
            os.makedirs(os.getcwd()+'/'+dump_dir, exist_ok=True)

            dump_op = '--cmd'+' '+os.environ.get('train_cmd')+' '+'--nj 1'+' '+'--do_delta'+' '+do_delta+ \
                    ' '+'data/'+x+'/feats.scp'+' '+'data/'+basis_dataset[0]+'/cmvn.ark'+' '+'exp/dump_feats/'+x+' '+dump_dir

            #if x=='train-clean' or x=='train-test':
            ret = subprocess.call(['dump.sh'+' '+dump_op], shell=True, timeout=None)

            #for rtask in recog_set:
            #    feat_recog_dir='dump/'+rtask+'/delta'+do_delta
            #    os.makedirs()
    else:
        print("skip! stage 1: Feature Generation")


    ## dict_path only exists in train-clean. Fixed!!
    dict_path = 'data/lang_1char/'+traindata+'_units.txt'

## stage 2: Dictionary and Json Data Preparation
    if stage <= 2:
        print('stage 2: Dictionary and Json Data Preparation')
        for x in stage2_dataset:
            ## make dictionary
            os.makedirs(os.getcwd()+'/data/lang_1char', exist_ok=True)
            curr = os.getcwd()
            opt = 'cut -f 2- -d" "'+' '+'|'+' '+'tr " " "\\n"'+' '+'|'+' '+'sort | LC_COLLATE="ko_KR.UTF-8" uniq | '+"grep -v -e '^\\s*$'"
                   # "awk '{{print $0 " " NR+1}}'"+' >> '+os.getcwd()+'/data/lang_1char/'+x+'_units.txt'
            cmd = 'python'+' '+curr+'/../../../utils/'+'text2token.py'+' '+'-s 1 -n 1 data/'+x+'/text'
            #proc1 = subprocess.Popen(['python', curr+'/../../../utils/'+'text2token.py', '-s 1', '-n 1', 'data/'+x+'/text'], stdout=subprocess.PIPE)
            #proc2 = subprocess.Popen(['cut', '-f 2-', '-d" "'], stdin=proc1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if x in basis_dataset:
                proc = subprocess.Popen(cmd+' | '+opt, shell=True, stdout=subprocess.PIPE)

                out, err = proc.communicate()
                lines = out.decode('utf-8').splitlines()

                with open(os.getcwd()+'/data/lang_1char/'+x+'_units.txt', 'w+', encoding='utf-8') as dicfile:
                    for index,dic in enumerate(lines, 2):
                        if index == 2:
                            dicfile.writelines('<unk>'+' '+'1'+'\n')

                        dicfile.writelines(str(dic)+' '+str(index)+'\n')

            #print('out: {0}'.format(lines))

            ## make json labels "espnet/utils/data2json.sh" <<-- common scripts

            dump_dir = 'dump/'+x+'/delta'+do_delta
            ## data/lang_1char/train-clean_units.txt exists only!!!
            data2json_op = ' --feat '+dump_dir+'/feats.scp'+' '+'data/'+x+' '+dict_path+' > '+dump_dir+'/data.json'
            subprocess.call([curr+'/../../../utils/data2json.sh'+data2json_op], shell=True)
    else:
        print("skip! stage 2: Dictionary and Json Data Preparation")

## stage 3: model training or transfer
    expdir = 'exp/' + expname

    if stage <= 3:
        print("start ASR training")
        ## training command && option

        # if preprocess_config=="":
        #     expname = traindata+'_'+backend+'_'+'train_transducer'
        # else:
        #     preconfig = preprocess_config.split('/')[-1].split('.')[0]
        #     expname = traindata+'_'+backend+'_'+'train_transducer_'+preconfig



        n_mini = '0'
        if os.path.exists(os.getcwd()+'/'+expdir+'/results/model.loss.best'):
            return 0

                #'--preprocess-conf '+preprocess_config+' '+ \
        os.makedirs(os.getcwd()+'/'+expdir, exist_ok=True)
        train_cmd = os.environ.get('cuda_cmd')+' '+'--gpu '+ngpu+' '+expdir+'/traing.log '
        train_option = 'asr_train.py '+ \
                '--config '+ train_config+' '+ \
                '--preprocess-conf '+preprocess_config+' '+ \
                '--ngpu '+ngpu+' '+ \
                '--backend '+backend+' '+ \
                '--outdir '+expdir+'/results '+ \
                '--tensorboard-dir '+'tensorboard/'+expname+' '+ \
                '--debugmode '+ debugmode+ ' '+ \
                '--dict '+ dict_path+ ' '+ \
                '--debugdir '+expdir+' '+ \
                '--minibatches '+n_mini+' '+ \
                '--beam-size 1 '+\
                '--verbose '+verbose+' '+ \
                '--report-cer '+ \
                '--report-wer ' + \
                '--train-json '+dump_dir_tr+'/data.json'+' '+ \
                '--valid-json '+dump_dir_dt+'/data.json'+' '\
                '--vgg_ichannels 64 64 128 128 ' + \
                '--resume ' + resume + ' '
        subprocess.call([train_cmd+train_option], shell=True)
    else:
        print("skip! stage 3: ASR Training")

    # if preprocess_config=="":
    #     expname = traindata+'_'+backend+'_'+'train_transducer'
    # else:
    #     preconfig = preprocess_config.split('/')[-1].split('.')[0]
    #     expname = traindata+'_'+backend+'_'+'train_transducer_'+preconfig

    main_expdir='exp/'+expname
## stage 4: Decoding
    if stage <= 4:
        print('state4: Start inference')
        nj = '1'
        #decode_config_dir=decode_config.split('/')[-1].split('.')[0]
        for rtask in recog_dataset:
            decode_config_dir=decode_config.split('/')[-1].split('.')[0]
            decode_dir='decode_'+rtask+'_'+decode_config_dir
            feat_recog_dir='dump/'+rtask+'/delta'+do_delta

            split_cmd='python'+' '+os.getcwd()+'/../../../utils/'+'splitjson.py '
            split_option='--parts '+nj+' '+feat_recog_dir+'/'+'data.json'

            subprocess.call([split_cmd+split_option], shell=True)

            decode_cmd= os.environ.get('decode_cmd')+' '+'JOB=1:'+nj+' '+main_expdir+'/'+decode_dir+'/log/decode.JOB.log '
            decode_option= 'asr_recog.py '+ \
                    '--config '+decode_config+' '+ \
                    '--ngpu '+'0'+' '+ \
                    '--backend '+'pytorch'+' '+ \
                    '--debugmode '+'1'+' '+ \
                    '--recog-json '+ feat_recog_dir+'/split'+nj+'utt/data.JOB.json'+' '+ \
                    '--result-label '+main_expdir+'/'+decode_dir+'/data.JOB.json'+' '+ \
                    '--model '+main_expdir+'/results/model.loss.best'

            subprocess.call([decode_cmd+decode_option], shell=True)

            score_cmd='score_sclite.sh '+'--wer false'+' '+main_expdir+'/'+decode_dir+' '+dict_path
            subprocess.call([score_cmd], shell=True)

    print('finished')

if __name__ == '__main__':

    main()