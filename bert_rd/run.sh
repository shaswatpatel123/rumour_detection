python ./main_2label.py ./unaugmented bert-base-uncased ./models/2label/unaugmented/bert > ./logger/bert_unaugmented
python ./main_2label.py ./augmented bert-base-uncased ./models/2label/augmented/bert > ./logger/bert_augmented
python ./main_2label.py ./improved bert-base-uncased ./models/2label/improved/bert > ./logger/bert_improved
