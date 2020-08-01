#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=2G #number of memery
#$ -o preprocess_kgcn.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M myuan7@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

module load dev/cmake/3.7.1/gcc-4.9.4



#source activate myspark

#Create an conda virtual environment called 'tensorflow'
conda create -n tensorflow python=3.6
#Activate the 'tensorflow' environment
source activate tensorflow
pip install tensorflow
pip install scikit-learn


cd Cluster_KGCN/metis-5.1.0
make config shared=1 prefix=~/.local/
make install
export METIS_DLL=~/.local/lib/libmetis.so

cd ..
pip install -r requirements.txt

cd src/
python3 preprocess.py -d music
python3 main.py
