import argparse, os, random
def run(dPath, mpath):
    datasetTypes = ['train','validation','test']
    dataRange = [(0, 0.9),(0.9, 0.905),(0.905, 1.0)]
    # datasetTypes = ['all']
    # dataRange = [(0, 1.0)]

    filePathList = None
    textList = None

    #loading metadata
    with open(mpath, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        tmp2 = [t.strip().split('|') for t in tmp]

        filePathList = [t[0].replace('.\\','').replace('./','').replace('\\','/') for t in tmp2]
        textList = [t[1] for t in tmp2]
        spkList = [t[2].strip() for t in tmp2]

    # random indexs for wave files
    numFiles = len(filePathList)
    indList = [ i for i in  range(numFiles)]
    random.shuffle(indList)

    # generate meta information
    for i, d in enumerate(datasetTypes):
        # make directorys
        targetPath = os.path.join('data',d)
        os.makedirs(targetPath,exist_ok=True)

        wavscpFile = open(os.path.join(targetPath, 'wav.scp'), 'w', encoding='utf-8')
        utt2spkFile = open(os.path.join(targetPath, 'utt2spk'), 'w', encoding='utf-8')
        textFile = open(os.path.join(targetPath, 'text'), 'w', encoding='utf-8')
        spkidFile = open(os.path.join(targetPath, 'spkid'), 'w', encoding='utf-8')

        # computing indexs
        sind, eind = int(dataRange[i][0]*numFiles), int(dataRange[i][1]*numFiles)
        targetIndexs = indList[sind:eind][:]
        targetIndexs.sort()
        #print(len(targetIndexs))

        # generating meta information
        wavscpMeta = []
        utt2spkMeta = []
        textMeta = []
        spkDic = {}
        for i, ti in enumerate(targetIndexs):
            filePath = os.path.join(dPath, filePathList[ti])
            text = textList[ti]
            spk = spkList[ti]
            spkidMeta = spkDic.get(spk)
            if(spkidMeta == None):
                spkDic[spk] = ['{} '.format(spk)]
                spkidMeta = spkDic.get(spk)

            wavscpMeta.append("{:08d} {}\n".format(i, filePath))
            utt2spkMeta.append("{:08d} {}\n".format(i, spk))
            textMeta.append("{:08d} {}\n".format(i, text))
            spkidMeta.append("{:08d} ".format(i))

        spkidmetas = []
        for k in spkDic.keys():
            spkidmetas += spkDic.get(k)
        # saving meta information
        wavscpFile.writelines(wavscpMeta)
        utt2spkFile.writelines(utt2spkMeta)
        textFile.writelines(textMeta)
        spkidFile.writelines(spkidMeta)

    pass

if __name__ == '__main__':
    """
    Purpose
        prepare metafiles for kaldiio
    Input
        dataset
        metadata.csv
        - file path | text
    Output
        data/train,test,val
        - wav.scp: (index) (filepath)\n
        - utt2spk: (index) (spkid or anonymous)\n
        - text: (index) (text)\n
        - spk2utt: (spkid) (index)1 (index)2 .... (index)end
    usage
        python prepare_dataset_for_kaldiio.py -d audio/projects/rnn-transducer-old/AIhub -m audio/projects/rnn-transducer-old/AIhub/metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_directory', type=str, default='',)
    parser.add_argument('-m', '--metadata_path', type=str, default='',
                        required=True, help='metadata paths')
    # parser.add_argument('--mode', type=str, default='',
    #                     required=True)

    args = parser.parse_args()
    # assert args.mode not in ['innerspk', 'interspk'], 'mode must be innerspk or interspk'

    run(args.dataset_directory, args.metadata_path)
    #run(args.dataset_directory, args.metadata_path, args.mode)