import matplotlib
import os
import gc
import warnings

warnings.filterwarnings('ignore')
import sys
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa, librosa.display, IPython.display as ipd
import json
from mutagen.mp3 import MP3
from statistics import mean, median
import noisereduce as no
import contextlib
import wave
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import contextlib
import boto3
import argparse
import time


# this is ideal if all your files are on local compute disk, otherwise, make it event driven

def uselocalfs():
    for root, dirs, files in os.walk("/home/ec2-user/SageMaker/birdsounds/dataset/audio/", topdown=True):
        if root == '/home/ec2-user/SageMaker/birdsounds/dataset/audio/':
            birds = dirs
    birds50 = []
    flist = []
    blist = []
    i50 = 0;
    for i, bird in enumerate(birds):
        for root, dirs, files in os.walk("/home/ec2-user/SageMaker/birdsounds/dataset/audio/" + bird):
            for file in files:
                if file.endswith(".mp3"):
                    blist.append(os.path.join(root, file))
        if len(blist) > 50:
            i50 = i50 + 1;
            birds50.append(bird)
            flist.append(blist)
        blist = []
    return blist, flist, birds50


def float_range(start, stop, step):
    import decimal
    while start <= stop:
        yield float(start)
        start += float(decimal.Decimal(step))


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def saveMel(y, directory):
    N_FFT = 1024  # Number of frequency bins for Fast Fourier Transform
    HOP_SIZE = 1024  # Number of audio frames between STFT columns
    SR = 44100  # Sampling frequency
    N_MELS = 30  # Mel band parameters
    WIN_SIZE = 1024  # number of samples in each STFT window
    WINDOW_TYPE = 'hann'  # the windowin function
    FEATURE = 'mel'  # feature representation

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ## describe the 3 parameters below

    HOP_SIZE = 1024
    N_MELS = 128
    FMIN = 1400
    S = librosa.feature.melspectrogram(y=y, sr=SR,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       fmin=FMIN,
                                       fmax=SR / 2)
    librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), fmin=FMIN, y_axis='linear')
    print(directory)
    fig.savefig(directory)
    fig.clear()
    ax.cla()
    plt.clf()
    plt.close('all')


def downloadfilefroms3(bucket, filepath, localpath="/tmp"):
    import boto3

    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, filepath, localpath + "/" + filepath.split("/")[-1])
        return localpath + "/" + filepath.split("/")[-1]

    except Exception as e:
        print(e)


def uploadfiletos3(localpath, bucket, objname):
    s3 = boto3.client("s3")
    s3.upload_file(localpath, bucket, objname)
    print("upload complete " + str(localpath))


def readWindowSize(bucket, obj="parameters/params.json"):
    try:

        jsonpath = downloadfilefroms3(bucket, obj)
        print(jsonpath)
        time.sleep(2)
        with open(jsonpath, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print("params file not found :" + str(e))
        return None


def processAllFiles(csvPath='/opt/ml/processing/input_data/labels.csv'):
    '''
    :param

    csvFile contains filenames and class.
    '''

    label_df = pd.read_csv(csvPath)
    basePath = '/opt/ml/processing/input_data/audio/audio/44100/'
    for index, row in label_df.iterrows():
        try:
            createmelspecs(basePath + row['filename'], subfolder=row['broadclass'])
        except Exception as e:
            print("file not found may be")

    return


def createmelspecs(objpath, minimum=1, stride=0, name=5, subfolder=None):
    print("received file {}".format(objpath))
    '''
    params=readWindowSize(bucket)
    #params={"desired":0.5}
    if params is None:
        print("issue with params file in :"+str(bucket))
        return None
    '''

    # these are hard coded for now, once working, turn it into functional args
    size = {'desired': 5,  # [seconds]  desired': float(params['desired'])
            'minimum': minimum,  # [seconds]
            'stride': stride,  # [seconds]
            'name': name
            }

    step = 1
    desired_ext = str(size['desired']).replace(".", "-")

    # path=downloadfilefroms3(bucket, objpath)
    path = objpath

    outputpath = "/opt/ml/processing/output_data/melspecs/" + subfolder + "/"
    print(outputpath)
    if step > 0:
        try:
            print("starting mfcc generation ")
            # directory = "/tmp/mels-2class/"
            directory = outputpath
            if not os.path.exists(directory):
                os.makedirs(directory)
            if not os.path.exists(directory + path.rsplit('/', 1)[1].replace(' ', '')[:-4] + "1_1.png"):
                try:
                    y, sr = librosa.load(path, sr=44100, mono=True)
                except Exception as e:
                    print("may be file not found or another reason")

                # getting duration in case you want a snapshot of the full spectrum instead of a sliding window
                # dur=librosa.get_duration(y=y, sr=sr)
                # step=dur*sr
                y = no.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)
                step = (size['desired'] - size['stride']) * sr
                nr = 0
                # for start, end in zip(range(0, len(y), step), range(size['desired'] * sr, len(y), step)):
                for start, end in zip(float_range(0, len(y), step), float_range(size['desired'] * sr, len(y), step)):
                    nr = nr + 1
                    print(nr)
                    print(start, end)
                    if end - start > size['minimum'] * sr:
                        melpath = path.rsplit('/', 1)[1]
                        melpath = directory + melpath.replace(' ', '')[:-4] + str(nr) + "_" + str(
                            nr) + "_" + desired_ext + ".png"
                        print(melpath)
                        # saveMel(y[start:end], melpath)
                        saveMel(y[int(start):int(end)], melpath)
                        # uploadfiletos3(melpath, bucket, outputpath+"/"+melpath.split("/")[-1])
                    else:
                        print("end-start is too small")
            else:
                print("path already exists, clear the container cache")
            pass

        except ZeroDivisionError as e:
            print("excepting zero division error")
    else:
        print("Error: Stride should be lower than desired length.")

        return


s3_client = boto3.client('s3')


def download_dir(prefix, local, bucket, client=s3_client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--bucket', metavar='b', dest="bucket", type=str,
    #                         help='name of the source bucket')
    # parser.add_argument('--csvpath', dest='csvpath', metavar="o", type=str,
    #                         help='csv path with labels and filenames')

    # args = parser.parse_args()

    # print(args)

    # processAllFiles(args.bucket, args.csvpath)

    processAllFiles()

