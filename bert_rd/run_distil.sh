python ./main_2label.py ./unaugmented distilbert-base-uncased ./models_distilbert/2label/unaugmented/distilbert > ./logger/distilbert_unaugmented
python ./main_2label.py ./augmented distilbert-base-uncased ./models_distilbert/2label/augmented/distilbert > ./logger/distilbert_augmented
python ./main_2label.py ./improved distilbert-base-uncased ./models_distilbert/2label/improved/distilbert > ./logger/distilbert_improved
