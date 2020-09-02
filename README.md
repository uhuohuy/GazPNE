# ZSL-PLT
## Basic description
This study proposes a zero-shot learning approach for place entity tagging from tweets, named ZSL-PLT, which does not assume any annotated sentences at training time. It fuses rule, gazetteer, and deep learning-based approaches to achieve the best performance among all. Specifically, we apply a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)-fused deep learning model called C-LSTM to derive a general place name classifier based on abundant positive examples (around 22 million) from gazetteers (i.e., OpenStreetMap and GeoNames) and negative examples (around 220 million) synthesized by rules. The classifier is then used to score n-gram segments of the tweet text and select the top none-overlapping candidates. We evaluate the approach on 4,500 disaster-related tweets, including around 9,500 place names from three targeted streams corresponding to the floods in Louisiana (the US), Houston (the US), and Chennai (India), respectively. We provide a comparison against several competitive baselines. The results show that the proposed approach improves the average F1-score from 0.81 for the best performing system to 0.87 (a 7\% increase).

The Architecture of ZSL-PLT is as follows:
![Screenshot](figure/workflow.jpg)

## Neural Classifier
The first step of ZSL-PLT is to train a model based on positive examples from gazetters and negative examples sythesized by rules.
### Training examples perparation
Several important data are needed and should be put in the data folder.

OSM data: Used gazetters include OpenStreetMap and Geonames. Specifically, two boundary boxes are choosed to select the osm name entitits from OSMNames (https://osmnames.org/download/). This file is too huge and shared through google drive.

The Geoname data: Two files are IN.txt and US.txt, which can be downloaded through (https://download.geonames.org/export/dump/). They corresponse to the data in the whole US and India areas, respectively.  

Two word embeddings: Goolge-embedding and Golve-embedding.

After got all the data, [rawTextProcessing.py](rawTextProcessing.py) can be used to extract the positive examples and negative examples from the data file above.
The extracted positive and negative examples are shared through google drive.

### Specific Word embedding
Specific Word embedding can be obtained by applying the word2vector algorithm on the positive examples. This can be done by [word2vec-garzeteer.py](word2vec-garzeteer.py).

We have also provided the trained specific Word embedding on google drive. It should be put in the data folder

### Model training
We apply the C-LSTM  model in classifying the place entities, which combines the CNN and LSTM to achieve the best of both. The topology of the network is depicted as follows:
![Screenshot](figure/architecture.jpg)
[Garzetter_sim_pre.py](Garzetter_sim_pre.py) is used to train a classification model based on the positive and negative examples.

## Place tagger from tweet texts
The three tweet data sets are avaliable here and they should be put under the data folder. The trained model in the last step can be then used to extract the place from the tweet data set by [model_test_json.py](model_test_json.py).

## Experimental results

![Screenshot](figure/597627196.jpg)


