./data_prep_3label.py ./all-rnr-annotated-threads ./data/pheme9/3label
./main_3label.py ./data/pheme9/3label/unaugmented ./data/pheme9/3label ./model/pheme9/3label/unaugmented/ GCN 100
./main_3label.py ./data/pheme9/3label/unaugmented ./data/pheme9/3label ./model/pheme9/3label/unaugmented/ GAT 100
./main_3label.py ./data/pheme9/3label/augmented ./data/pheme9/3label ./model/pheme9/3label/augmented/ GCN 100
./main_3label.py ./data/pheme9/3label/augmented ./data/pheme9/3label ./model/pheme9/3label/augmented/ GAT 100
./main_3label.py ./data/pheme9/3label/improved ./data/pheme9/3label ./model/pheme9/3label/improved/ GCN 100
./main_3label.py ./data/pheme9/3label/improved ./data/pheme9/3label ./model/pheme9/3label/improved/ GAT 100

./data_prep_2label.py ./all-rnr-annotated-threads ./data/pheme9/2label
./main_2label.py ./data/pheme9/2label/unaugmented ./data/pheme9/2label ./model/pheme9/2label/unaugmented/ GCN 100
./main_2label.py ./data/pheme9/2label/unaugmented ./data/pheme9/2label ./model/pheme9/2label/unaugmented/ GAT 100
./main_2label.py ./data/pheme9/2label/augmented ./data/pheme9/2label ./model/pheme9/2label/augmented/ GCN 100
./main_2label.py ./data/pheme9/2label/augmented ./data/pheme9/2label ./model/pheme9/2label/augmented/ GAT 100
./main_2label.py ./data/pheme9/2label/improved ./data/pheme9/2label ./model/pheme9/2label/improved/ GCN 100
./main_2label.py ./data/pheme9/2label/improved ./data/pheme9/2label ./model/pheme9/2label/improved/ GAT 100