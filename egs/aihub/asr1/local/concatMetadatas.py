import argparse
import os
import random
import numpy as np

def concatMetas(srcNames, srcMetas, targetMeta, samplingType = 'original'):
    """
    :param srcNames:
    :param srcMetas:
    :param targetMeta:
    :param samplingType: original, upSampling, downSampling
    :return:
    """
    assert samplingType in ['original', 'upSampling', 'downSampling', '10percent']
    lenMetas = []
    for i in range(len(srcNames)):
        srcMeta = srcMetas[i]
        f = open(srcMeta,'r',encoding='utf-8')
        lenMetas.append(len(f.readlines()))
        f.close()
    maxLenMetas = max(lenMetas)
    minLenMetas = min(lenMetas)
    metas = []
    for i in range(len(srcNames)):
        srcName = srcNames[i]
        srcMeta = srcMetas[i]
        f = open(srcMeta,'r',encoding='utf-8')
        meta = [m.replace('wavs',srcName) for m in f.readlines()]
        if(samplingType == 'upSampling'):
            meta = upSampling(meta, maxLenMetas)
        elif(samplingType == 'downSampling'):
            meta = downSampling(meta, minLenMetas)
        elif(samplingType == '10percent'):
            if(len(meta) == maxLenMetas):
                meta = upSampling(meta, maxLenMetas)
            else:
                p10 = int(float(maxLenMetas)*0.1)
                if (len(meta) <= p10):
                    meta = upSampling(meta, p10)
                else:
                    meta = downSampling(meta,p10)

        metas += meta
        print(srcName, len(meta))
        f.close()

    f = open(targetMeta,'w',encoding='utf-8')
    f.writelines(metas)
    #print(lenMetas, len(metas))
    f.close()

def upSampling(metas, maxnum):
    numMetas = len(metas)
    assert numMetas <= maxnum
    metas_ = []
    random.shuffle(metas)

    val = maxnum/numMetas
    if(val ==1.0):
        return metas
    nRoof = int(np.ceil(val))
    afterDecimalPoint = val - np.floor(val)
    remainder = int(numMetas*afterDecimalPoint)
    for i in range(nRoof):
           if(i == nRoof - 1):
               metas_ += metas[:remainder]
           else:
               metas_ += metas
    metas_.sort()
    return metas_

def downSampling(metas, minNum):
    numMetas = len(metas)
    assert numMetas >= minNum
    metas_ = []
    random.shuffle(metas)

    # val = minNum/numMetas
    # afterDecimalPoint = val - np.floor(val)
    # remainder = int(numMetas*afterDecimalPoint)
    metas_ = metas[:minNum]
    metas_.sort()
    return metas_

if __name__ == '__main__':
    """
        usage
        python concatMetadatas.py --baseDir /audio/dataset --srcs 16000_sample/gcp,16000_sample/kakao,16000_sample/netmarble  --metas metadata_spk.csv,metadata_spk.csv,metadata_spk.csv
        python concatMetadatas.py --baseDir /audio/dataset --srcs AIhub,NIKL,16000_sample/gcp,16000_sample/kakao,16000_sample/netmarble  --metas metadata_spk.csv,metadata_spk.csv,metadata_spk.csv,metadata_spk.csv,metadata_spk.csv
        python concatMetadatas.py --baseDir /audio/dataset --srcs AIhub,NIKL,16000_sample/gcp,16000_sample/kakao,16000_sample/netmarble  --metas metadata_spk.csv,metadata_spk.csv,metadata_spk.csv,metadata_spk.csv,metadata_spk.csv --target metadata_up.csv --sampling upSampling
        python concatMetadatas.py --baseDir /audio/dataset --srcs AIhub,NIKL,gcp_kakao_netmarble  --metas metadata_spk.csv,metadata_spk.csv,metadata.csv --target metadata_up.csv --sampling upSampling
        python concatMetadatas.py --baseDir /audio/dataset --srcs AIhub,NIKL,gcp_kakao_netmarble  --metas metadata_spk.csv,metadata_spk.csv,metadata.csv --target metadata_10.csv --sampling 10percent
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseDir', type=str,
                        help='Parent folder containing the dataset folders')
    parser.add_argument('--srcs', type=str,
                        help='the dataset #1,2, ..., N')
    parser.add_argument('--metas', type=str,
                        help='the file name of metadata1 in the dataset #1,2, ...,N')
    parser.add_argument('--sampling', type=str,
                        default='original', help='')
    parser.add_argument('--target', type=str,
                        default='metadata.csv',help='')
    args = parser.parse_args()

    # preparing a target dir
    srcs = args.srcs.split(',')
    metas = args.metas.split(',')
    srcNames = [src.split('/')[-1] for src in srcs]
    srcMetas = []
    targetDir = os.path.join(args.baseDir, '_'.join(srcNames))
    os.makedirs(targetDir, exist_ok=True)

    # preparing parameters
    for i in range(len(srcs)):
        srcDir = os.path.join(args.baseDir, srcs[i])
        srcMeta = os.path.join(srcDir, metas[i])
        srcMetas.append(srcMeta)
        symbolicDir = os.path.join(targetDir, srcNames[i])
        print(srcDir, srcMeta, symbolicDir)
        cmd1 = "rm -r {symbolicDir}; ln -s {srcDir}/wavs {symbolicDir}".format(srcDir=srcDir,
                                                                               symbolicDir=symbolicDir)
        os.system(cmd1)
    targetMeta = os.path.join(targetDir, args.target)

    # concatinating metadatas
    concatMetas(srcNames, srcMetas, targetMeta, args.sampling)

    pass
