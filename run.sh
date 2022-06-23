pip install emoji
pip install transformers
pip install tabulate
pip install nlpaug
pip install sentencepiece
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html

wget https://ndownloader.figshare.com/articles/6392078/versions/1
unzip ./1
tar -xf ./PHEME_veracity.tar.bz2

mkdir logger

./data_prep_3label.py ./all-rnr-annotated-threads ./data/pheme9/3label > ./logger/3labelprep.txt
./main_3label.py ./data/pheme9/3label/unaugmented ./data/pheme9/3label ./model/pheme9/3label/unaugmented/ GCN 100 > ./logger/unaugmented_3label_gcn.txt
./main_3label.py ./data/pheme9/3label/unaugmented ./data/pheme9/3label ./model/pheme9/3label/unaugmented/ GAT 100 > ./logger/unaugmented_3label_gat.txt
./main_3label.py ./data/pheme9/3label/augmented ./data/pheme9/3label ./model/pheme9/3label/augmented/ GCN 100 > ./logger/augmented_3label_gcn.txt
./main_3label.py ./data/pheme9/3label/augmented ./data/pheme9/3label ./model/pheme9/3label/augmented/ GAT 100 > ./logger/augmented_3label_gat.txt
./main_3label.py ./data/pheme9/3label/improved ./data/pheme9/3label ./model/pheme9/3label/improved/ GCN 100 > ./logger/improved_3label_gcn.txt
./main_3label.py ./data/pheme9/3label/improved ./data/pheme9/3label ./model/pheme9/3label/improved/ GAT 100 > ./logger/improved_3label_gat.txt

./data_prep_2label.py ./all-rnr-annotated-threads ./data/pheme9/2label > ./logger/2labelprep.txt
./main_2label.py ./data/pheme9/2label/unaugmented ./data/pheme9/2label ./model/pheme9/2label/unaugmented/ GCN 100 > ./logger/unaugmented_2label_gcn.txt
./main_2label.py ./data/pheme9/2label/unaugmented ./data/pheme9/2label ./model/pheme9/2label/unaugmented/ GAT 100 > ./logger/unaugmented_2label_gat.txt
./main_2label.py ./data/pheme9/2label/augmented ./data/pheme9/2label ./model/pheme9/2label/augmented/ GCN 100 > ./logger/augmented_2label_gcn.txt
./main_2label.py ./data/pheme9/2label/augmented ./data/pheme9/2label ./model/pheme9/2label/augmented/ GAT 100 > ./logger/augmented_2label_gat.txt
./main_2label.py ./data/pheme9/2label/improved ./data/pheme9/2label ./model/pheme9/2label/improved/ GCN 100 > ./logger/improved_2label_gcn.txt
./main_2label.py ./data/pheme9/2label/improved ./data/pheme9/2label ./model/pheme9/2label/improved/ GAT 100 > ./logger/improved_2label_gat.txt