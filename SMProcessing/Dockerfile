FROM ubuntu:16.04

# Contact me for help!
MAINTAINER rkadiy@amazon.com

# The WORKDIR instruction sets the working directory for any RUN, CMD, ENTRYPOINT,
# COPY and ADD instructions that follow it in the Dockerfile. If the WORKDIR doesn’t
# exist, it will be created even if it’s not used in any subsequent Dockerfile instruction.
WORKDIR /home

#####################################################################
# Make required directories
#####################################################################

# Now create some diectories to store files and scripts
RUN mkdir code
RUN mkdir code/samples
RUN mkdir code/samples/libROSA
RUN mkdir code/samples/TensorFlow
RUN mkdir workspace

#####################################################################
# Install System tools needed
#####################################################################

RUN apt-get update \
  && apt-get install -y build-essential \
  wget \
  unzip \
  bzip2 \
  software-properties-common \
  python-setuptools

# Music libraries
RUN apt-get install -y libasound-dev \
   portaudio19-dev \
   libportaudio2 \
   libportaudiocpp0 \
   ffmpeg \
   libavcodec-extra

#####################################################################
# Install Anaconda (python 2.7 version)
#####################################################################

# Change to the code (aka software directory).
WORKDIR /home/code

# Now install Anaconda to manage the python environment. This installs numpy, scipy, matplotlib,
# jupyter etc. Type conda list inside the built docker container, to view the python modules already
# installed.
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b -p /home/code/anaconda
RUN rm Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /home/code/anaconda/bin:$PATH

#####################################################################
# Update Anaconda (extra data science tools)
#####################################################################

# Python packages from conda. Here libgcc is needed to run XGBoost,
# whilst GCC is needed for neural network based tools.
RUN apt-get install -y python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN conda install -c anaconda pillow
# RUN conda install -c anaconda opencv
RUN python3.7 -m pip install opencv-python
RUN conda install --channel=numba llvmlite
RUN python3.7 -m pip install soundfile
RUN python3.7 -m pip install mutagen
RUN python3.7 -m pip install noisereduce
RUN python3.7 -m pip install cffi==1.11.5


# Music tools
RUN conda config --append channels conda-forge
RUN conda config --append channels mutirri
RUN conda config --append channels auto

RUN conda install -c conda-forge ffmpeg
RUN conda install -c auto pydub=0.23.1
RUN conda install -c conda-forge librosa
RUN conda install -c mutirri pyaudio=0.2.11

# Some machine learning tools
RUN conda update scikit-learn
RUN conda update conda



#####################################################################
# Install CPU version of Tensor Flow
#####################################################################


# IPython
EXPOSE 8888

WORKDIR /home/testcode

COPY melspecs.py /home/testcode
# Get samples
WORKDIR /home/workspace/samples/libROSA
RUN wget https://raw.githubusercontent.com/scienceguyrob/Docker/master/Images/music/samples/libROSA/sample.m4a
RUN wget https://raw.githubusercontent.com/scienceguyrob/Docker/master/Images/music/samples/libROSA/LibROSA_Demo.ipynb


#####################################################################
# Finally...
#####################################################################

# Ensure the OS is updated.
RUN apt-get update
RUN python3.7 -m pip install awscli
RUN python3 -m pip install --user boto3
RUN apt-get install -y vim 

# Define the command that will be exectuted when docker runs the container.
WORKDIR /home/testcode
ENTRYPOINT python3 melspecs.py --bucket ${INPUT_BUCKET} --objpath "${OBJ_PATH}"
