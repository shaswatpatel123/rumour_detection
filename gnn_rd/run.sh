set -e

# pip install emoji
# pip install transformers
# pip install tabulate
# pip install nlpaug
# pip install sentencepiece
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html

wget https://ndownloader.figshare.com/articles/6392078/versions/1
unzip ./1
tar -xf ./PHEME_veracity.tar.bz2

rm -rf ./PHEME_veracity.tar.bz2
rm -rf ./1

wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/6453753/phemernrdataset.tar.bz2
tar -xf ./phemernrdataset.tar.bz2
rm -rf phemernrdataset.tar.bz2

wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xf ./gdrive_2.1.1_linux_386.tar.gz
rm -rf gdrive_2.1.1_linux_386.tar.gz

mkdir -p logger

# python ./tweet_to_feature.py ./all-rnr-annotated-threads ./

python ./data_prep_3label.py ./all-rnr-annotated-threads ./data/pheme9/3label > ./logger/3labelprep.txt
python ./main_3label.py ./data/pheme9/3label/unaugmented ./model/pheme9/3label/unaugmented/ GCN 100 ./data/pheme9/3label/pheme ./ > ./logger/unaugmented_3label_gcn.txt
python ./main_3label.py ./data/pheme9/3label/unaugmented ./model/pheme9/3label/unaugmented/ GAT 100 ./data/pheme9/3label/pheme ./ > ./logger/unaugmented_3label_gat.txt
python ./main_3label.py ./data/pheme9/3label/augmented ./model/pheme9/3label/augmented/ GCN 100 ./data/pheme9/3label/pheme ./ > ./logger/augmented_3label_gcn.txt
python ./main_3label.py ./data/pheme9/3label/augmented ./model/pheme9/3label/augmented/ GAT 100 ./data/pheme9/3label/pheme ./ > ./logger/augmented_3label_gat.txt
python ./main_3label.py ./data/pheme9/3label/improved ./model/pheme9/3label/improved/ GCN 100 ./data/pheme9/3label/pheme ./ > ./logger/improved_3label_gcn.txt
python ./main_3label.py ./data/pheme9/3label/improved ./model/pheme9/3label/improved/ GAT 100 ./data/pheme9/3label/pheme ./ > ./logger/improved_3label_gat.txt

python ./data_prep_2label.py ./all-rnr-annotated-threads ./data/pheme9/2label > ./logger/2labelprep.txt
python ./main_2label.py ./data/pheme9/2label/unaugmented ./model/pheme9/2label/unaugmented/ GCN 100 ./data/pheme9/2label/pheme ./ > ./logger/unaugmented_2label_gcn.txt
python ./main_2label.py ./data/pheme9/2label/unaugmented ./model/pheme9/2label/unaugmented/ GAT 100 ./data/pheme9/2label/pheme ./ > ./logger/unaugmented_2label_gat.txt
python ./main_2label.py ./data/pheme9/2label/augmented ./model/pheme9/2label/augmented/ GCN 100 ./data/pheme9/2label/pheme ./ > ./logger/augmented_2label_gcn.txt
python ./main_2label.py ./data/pheme9/2label/augmented ./model/pheme9/2label/augmented/ GAT 100 ./data/pheme9/2label/pheme ./ > ./logger/augmented_2label_gat.txt
python ./main_2label.py ./data/pheme9/2label/improved ./model/pheme9/2label/improved/ GCN 100 ./data/pheme9/2label/pheme ./ > ./logger/improved_2label_gcn.txt
python ./main_2label.py ./data/pheme9/2label/improved ./model/pheme9/2label/improved/ GAT 100 ./data/pheme9/2label/pheme ./ > ./logger/improved_2label_gat.txt
