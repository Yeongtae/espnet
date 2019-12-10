import argparse, os, random
def run(dPath, mpath):
    datasetTypes = ['train','validation','test']
    dataRange = [(0, 0.9),(0.9, 0.95),(0.95, 1.0)]

    filePathList = None
    textList = None

    #loading metadata
    with open(mpath, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        tmp2 = [t.strip().split('|') for t in tmp]

        filePathList = [t[0].replace('.\wav','wav') for t in tmp2]
        textList = [t[1] for t in tmp2]

    # 연산 준비작업
    numFiles = len(filePathList)
    indList = [ i for i in  range(numFiles)]
    random.shuffle(indList)

    # 데이터셋 별 메타 정보 생성
    for i, d in enumerate(datasetTypes):
        # make directorys
        targetPath = os.path.join('data',d)
        os.makedirs(targetPath,exist_ok=True)

        wavscpFile = open(os.path.join(targetPath, 'wav.scp'), 'w', encoding='utf-8')
        utt2spkFile = open(os.path.join(targetPath, 'utt2spk'), 'w', encoding='utf-8')
        textFile = open(os.path.join(targetPath, 'text'), 'w', encoding='utf-8')
        spkidFile = open(os.path.join(targetPath, 'spkid'), 'w', encoding='utf-8')

        # 데이터 비율 선택
        sind, eind = int(dataRange[i][0]*numFiles), int(dataRange[i][1]*numFiles)

        # 파일리스트 준비, 소팅
        targetIndexs = indList[sind:eind]
        targetIndexs.sort()
        #print(len(targetIndexs))

        # index는? 그냥 쓰자, 그래도 디버깅 가능하자나

        # 그걸로 메타 4개 만들고
        wavscpMeta = []
        utt2spkMeta = []
        textMeta = []
        spkidMeta = ['anonymous ']
        for ti in targetIndexs:
            filePath = os.path.join(dPath, filePathList[ti])
            text = textList[ti]
            wavscpMeta.append("{:08d} {}\n".format(ti, filePath))
            utt2spkMeta.append("{:08d} {}\n".format(ti, 'anonymous'))
            textMeta.append("{:08d} {}\n".format(ti, text))
            spkidMeta.append("{:08d} ".format(ti))

        # 파일로 세이브
        wavscpFile.writelines(wavscpMeta)
        utt2spkFile.writelines(utt2spkMeta)
        textFile.writelines(textMeta)
        spkidFile.writelines(spkidMeta)

    pass

if __name__ == '__main__':
    """
    개발 목적
        espnet run.sh 상 0 번 작업
        kaldiIO 가 처리할 수 있는 데이터 포멧으로 metafile들을 준비함
    입력
        LJSpeech 스타일의 metadata를 포함한 dataset
        metadata.csv
        ㄴ file path | text
    출력
        data/train,test,val
        ㄴ wav.scp: (index) (filepath)\n
        ㄴ utt2spk: (index) (spkid or anonymous)\n
        ㄴ text: (index) (text)\n
        ㄴ spk2utt: (spkid) (index)1 (index)2 .... (index)end
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