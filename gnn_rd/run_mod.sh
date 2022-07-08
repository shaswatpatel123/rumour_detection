set -e
mkdir -p logger30
mkdir -p logger30/pheme9
mkdir -p logger30/pheme5

# python ./data_prep_3label.py ./data/all-rnr-annotated-threads ./data30/pheme9/3label ./ 0.30 0.30 > ./logger30/pheme9/3labelprep.txt

python ./main_3label.py ./data30/pheme9/3label/unaugmented ./model30/pheme9/3label/unaugmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/unaugmented_3label_gcn.txt
python ./main_3label.py ./data30/pheme9/3label/augmented ./model30/pheme9/3label/augmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/augmented_3label_gcn.txt
python ./main_3label.py ./data30/pheme9/3label/improved ./model30/pheme9/3label/improved/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/improved_3label_gcn.txt
python ./main_3label.py ./data30/pheme9/3label/unaugmented ./model30/pheme9/3label/unaugmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/unaugmented_3label_gat.txt
python ./main_3label.py ./data30/pheme9/3label/augmented ./model30/pheme9/3label/augmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/augmented_3label_gat.txt
python ./main_3label.py ./data30/pheme9/3label/improved ./model30/pheme9/3label/improved/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/improved_3label_gat.txt

python ./data_prep_2label.py ./data/all-rnr-annotated-threads ./data30/pheme9/2label ./ 0.30 0.30 > ./logger30/pheme9/2labelprep.txt

python ./main_2label.py ./data30/pheme9/2label/unaugmented ./model30/pheme9/2label/unaugmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/unaugmented_2label_gcn.txt
python ./main_2label.py ./data30/pheme9/2label/augmented ./model30/pheme9/2label/augmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/augmented_2label_gcn.txt
python ./main_2label.py ./data30/pheme9/2label/improved ./model30/pheme9/2label/improved/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/improved_2label_gcn.txt
python ./main_2label.py ./data30/pheme9/2label/unaugmented ./model30/pheme9/2label/unaugmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/unaugmented_2label_gat.txt
python ./main_2label.py ./data30/pheme9/2label/augmented ./model30/pheme9/2label/augmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/augmented_2label_gat.txt
python ./main_2label.py ./data30/pheme9/2label/improved ./model30/pheme9/2label/improved/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger30/pheme9/improved_2label_gat.txt

python ./data_prep_2label.py ./data/pheme-rnr-dataset ./data30/pheme5/2label ./ 0.30 0.30 > ./logger30/pheme5/2labelprep.txt

python ./main_2label.py ./data30/pheme5/2label/unaugmented ./model30/pheme5/2label/unaugmented/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/unaugmented_2label_gcn.txt
python ./main_2label.py ./data30/pheme5/2label/augmented ./model30/pheme5/2label/augmented/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/augmented_2label_gcn.txt
python ./main_2label.py ./data30/pheme5/2label/improved ./model30/pheme5/2label/improved/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/improved_2label_gcn.txt
python ./main_2label.py ./data30/pheme5/2label/unaugmented ./model30/pheme5/2label/unaugmented/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/unaugmented_2label_gat.txt
python ./main_2label.py ./data30/pheme5/2label/augmented ./model30/pheme5/2label/augmented/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/augmented_2label_gat.txt
python ./main_2label.py ./data30/pheme5/2label/improved ./model30/pheme5/2label/improved/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger30/pheme5/improved_2label_gat.txt

mkdir -p logger70
mkdir -p logger70/pheme9
mkdir -p logger70/pheme5

python ./data_prep_3label.py ./data/all-rnr-annotated-threads ./data70/pheme9/3label ./ 0.70 0.70 > ./logger70/pheme9/3labelprep.txt

python ./main_3label.py ./data70/pheme9/3label/unaugmented ./model70/pheme9/3label/unaugmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/unaugmented_3label_gcn.txt
python ./main_3label.py ./data70/pheme9/3label/augmented ./model70/pheme9/3label/augmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/augmented_3label_gcn.txt
python ./main_3label.py ./data70/pheme9/3label/improved ./model70/pheme9/3label/improved/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/improved_3label_gcn.txt
python ./main_3label.py ./data70/pheme9/3label/unaugmented ./model70/pheme9/3label/unaugmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/unaugmented_3label_gat.txt
python ./main_3label.py ./data70/pheme9/3label/augmented ./model70/pheme9/3label/augmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/augmented_3label_gat.txt
python ./main_3label.py ./data70/pheme9/3label/improved ./model70/pheme9/3label/improved/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/improved_3label_gat.txt

python ./data_prep_2label.py ./data/all-rnr-annotated-threads ./data70/pheme9/2label ./ 0.70 0.70 > ./logger70/pheme9/2labelprep.txt

python ./main_2label.py ./data70/pheme9/2label/unaugmented ./model70/pheme9/2label/unaugmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/unaugmented_2label_gcn.txt
python ./main_2label.py ./data70/pheme9/2label/augmented ./model70/pheme9/2label/augmented/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/augmented_2label_gcn.txt
python ./main_2label.py ./data70/pheme9/2label/improved ./model70/pheme9/2label/improved/ GCN 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/improved_2label_gcn.txt
python ./main_2label.py ./data70/pheme9/2label/unaugmented ./model70/pheme9/2label/unaugmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/unaugmented_2label_gat.txt
python ./main_2label.py ./data70/pheme9/2label/augmented ./model70/pheme9/2label/augmented/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/augmented_2label_gat.txt
python ./main_2label.py ./data70/pheme9/2label/improved ./model70/pheme9/2label/improved/ GAT 100 ./data/all-rnr-annotated-threads ./ > ./logger70/pheme9/improved_2label_gat.txt

python ./data_prep_2label.py ./data/pheme-rnr-dataset ./data70/pheme5/2label ./ 0.70 0.70 > ./logger70/pheme5/2labelprep.txt

python ./main_2label.py ./data70/pheme5/2label/unaugmented ./model70/pheme5/2label/unaugmented/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/unaugmented_2label_gcn.txt
python ./main_2label.py ./data70/pheme5/2label/augmented ./model70/pheme5/2label/augmented/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/augmented_2label_gcn.txt
python ./main_2label.py ./data70/pheme5/2label/improved ./model70/pheme5/2label/improved/ GCN 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/improved_2label_gcn.txt
python ./main_2label.py ./data70/pheme5/2label/unaugmented ./model70/pheme5/2label/unaugmented/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/unaugmented_2label_gat.txt
python ./main_2label.py ./data70/pheme5/2label/augmented ./model70/pheme5/2label/augmented/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/augmented_2label_gat.txt
python ./main_2label.py ./data70/pheme5/2label/improved ./model70/pheme5/2label/improved/ GAT 100 ./data/pheme-rnr-dataset ./ > ./logger70/pheme5/improved_2label_gat.txt