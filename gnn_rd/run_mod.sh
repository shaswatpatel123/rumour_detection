set -e
# mkdir -p logger

# +
# python ./data_prep_3label.py ./all-rnr-annotated-threads ./data/pheme5/3label > ./logger/3labelprep.txt
# python ./main_3label.py ./data/pheme5/3label/unaugmented ./model/pheme5/3label/unaugmented/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/unaugmented_3label_gcn.txt
# python ./main_3label.py ./data/pheme5/3label/augmented ./model/pheme5/3label/augmented/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/augmented_3label_gcn.txt
# python ./main_3label.py ./data/pheme5/3label/improved ./model/pheme5/3label/improved/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/improved_3label_gcn.txt
# python ./main_3label.py ./data/pheme5/3label/unaugmented ./model/pheme5/3label/unaugmented/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/unaugmented_3label_gat.txt
# python ./main_3label.py ./data/pheme5/3label/augmented ./model/pheme5/3label/augmented/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/augmented_3label_gat.txt
# python ./main_3label.py ./data/pheme5/3label/improved ./model/pheme5/3label/improved/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/improved_3label_gat.txt
# -

# python ./data_prep_2label.py ./all-rnr-annotated-threads ./data/pheme5/2label > ./logger/2labelprep.txt
python ./main_2label.py ./data/pheme5/2label/unaugmented ./model/pheme5/2label/unaugmented/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/unaugmented_2label_gcn.txt
python ./main_2label.py ./data/pheme5/2label/augmented ./model/pheme5/2label/augmented/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/augmented_2label_gcn.txt
python ./main_2label.py ./data/pheme5/2label/improved ./model/pheme5/2label/improved/ GCN 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/improved_2label_gcn.txt
python ./main_2label.py ./data/pheme5/2label/unaugmented ./model/pheme5/2label/unaugmented/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/unaugmented_2label_gat.txt
python ./main_2label.py ./data/pheme5/2label/augmented ./model/pheme5/2label/augmented/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/augmented_2label_gat.txt
python ./main_2label.py ./data/pheme5/2label/improved ./model/pheme5/2label/improved/ GAT 100 ./all-rnr-annotated-threads ./ > ./logger/pheme5/improved_2label_gat.txt
