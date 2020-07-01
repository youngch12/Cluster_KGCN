#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=4G #number of memery
#$ -o preprocess.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M myuan7@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

#source activate myspark

#Create an conda virtual environment called 'tensorflow'
conda create -n tensorflow python=3.6
#Activate the 'tensorflow' environment
source activate tensorflow
pip install tensorflow
pip install scikit-learn


cd KGCN-test/src
python3 preprocess.py -d movie
python3 main.py
