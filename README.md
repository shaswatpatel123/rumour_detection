# Rumour detection using graph neural network and oversampling in benchmark Twitter dataset
This repository contains the code for our paper "Rumour detection using graph neural network and oversampling in benchmark Twitter dataset". 

## BERT-based rumour detection pipeline
Our first paper "Leveraging User Comments in Tweets for Rumor Detection" uses comments and BERT(BERT, RoBERTa, ALBERT, and DistilBERT) for classifying tweets into rumour/non-rumour. We extended the pipeline to include multi-label classification: true/false/unverifed.

The data preparation file merges the source and comments in an array which is then feed to the models as a long-sequence.

## GNN-based rumour detection pipeline
Our second paper "Rumour detection using graph neural network and oversampling in benchmark Twitter dataset" uses BERT to extract textual embeddings, and GNN to capture spatial relations and propogation information.

The data preparation files creates: (i) adjacency list, and (ii) node-level features(text). It is easy to modify and reused by the community for future research.

Both the techniques also include our new data augmentation pipeline to oversample the under-represented classes in the dataset which has shown significant improvement. We also include pipeline for <b>early rumour detection</b>. We have trained and tested our approach on PHEME5 and PHEME9 datasets.

Feel free to reach out to us for any help via email or issue.

# Cite
```
@misc{patel2022rumourdetectionusinggraph,
      title={Rumour detection using graph neural network and oversampling in benchmark Twitter dataset}, 
      author={Shaswat Patel and Prince Bansal and Preeti Kaur},
      year={2022},
      eprint={2212.10080},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2212.10080}, 
}

@InProceedings{10.1007/978-981-16-2597-8_8,
author="Patel, Shaswat
and Shah, Binil
and Kaur, Preeti",
editor="Khanna, Ashish
and Gupta, Deepak
and Bhattacharyya, Siddhartha
and Hassanien, Aboul Ella
and Anand, Sameer
and Jaiswal, Ajay",
title="Leveraging User Comments in Tweets for Rumor Detection",
booktitle="International Conference on Innovative Computing and Communications",
year="2022",
publisher="Springer Singapore",
address="Singapore",
pages="87--99",
abstract="A novel technique is presented in this paper to detect rumors from tweets. Rumor is unverified information at the time of posting. To detect rumors in the tweets, we use transformer models BERT, RoBERTA, ALBERT, and DistilBERT. These techniques perform the feature extractor for input sequence consisting of source tweet and user comments on the source tweet. The key insight is that by understanding the context of the source tweet and the user comments the models can successfully classify the source tweet into rumor or non-rumor. This is based on the fact that users on social media sites try to classify any new information into rumor and non-rumor collectively by using comments. Our approach was able to produce better precision, recall, and F1 score over the state-of-the-art classifier that uses Conditional Random Fields (CRFs) to learn the context during the event.",
isbn="978-981-16-2597-8"
}


```
