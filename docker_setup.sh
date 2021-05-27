#!/usr/bin/env bash
# A shell script to initialize a new docker container that will host jupyter notebooks with the conda environment.
# Many thanks to Jas Bali for making this so much better than what I originally had here. -Ben
set -x
port=8887
dataset_repo_folder="$(mktemp -d)/tmp-datasets-folder"
dataset_folder="$dataset_repo_folder/datasets"
final_dataset_folder="$PWD/notebooks/TCPD/datasets"
mkdir -p $dataset_repo_folder

echo "Cloning datasets into folder: $dataset_repo_folder"
git clone https://github.com/alan-turing-institute/TCPD $dataset_repo_folder

rm -r $final_dataset_folder
echo "Copying datasets from $dataset_folder into $final_dataset_folder"
mkdir -p $final_dataset_folder
cp -r "$dataset_folder/" "$final_dataset_folder/"

echo "Starting Jupyter notebooks"s

docker run -i --name=MLEngineeringInAction \
-v $(PWD)/notebooks:/opt/notebooks -t \
-p $port:$port continuumio/anaconda3 bin/bash \
-c "/opt/conda/bin/conda install jupyter -y --quiet && \
   /opt/conda/bin/conda install -c conda-forge hyperopt=0.2.5 -y --quiet && \
   mkdir -p /opt/notebooks && \
   /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks \
   --ip='*' --port=$port --no-browser --allow-root"
