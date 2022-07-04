set -e
# python ./data_prep_3label.py
# mkdir -p logger
python ./main_3label.py ./data/unaugmented ./data/unaugmented ./model/unaugmented GCN 300 ./rumoureval2019 > ./logger/unaugmented_gcn.txt
python ./main_3label.py ./data/unaugmented ./data/unaugmented ./model/unaugmented GAT 300 ./rumoureval2019 > ./logger/unaugmented_gat.txt
python ./main_3label.py ./data/augmented ./data/unaugmented ./model/augmented GCN 300 ./rumoureval2019 > ./logger/augmented_gcn.txt
python ./main_3label.py ./data/augmented ./data/unaugmented ./model/augmented GAT 300 ./rumoureval2019 > ./logger/augmented_gat.txt
python ./main_3label.py ./data/improved ./data/unaugmented ./model/improved GCN 300 ./rumoureval2019 > ./logger/improved_gcn.txt
python ./main_3label.py ./data/improved ./data/unaugmented ./model/improved GAT 300 ./rumoureval2019 > ./logger/improved_gat.txt
