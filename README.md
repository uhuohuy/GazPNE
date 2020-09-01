# ZSL-PLT
## Basic description
This study proposes a zero-shot learning approach for place entity tagging from tweets, named ZSL-PLT, which does not assume any annotated sentences at training time. It fuses rule, gazetteer, and deep learning-based approaches to achieve the best performance among all. Specifically, we apply a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)-fused deep learning model called C-LSTM to derive a general place name classifier based on abundant positive examples (around 22 million) from gazetteers (i.e., OpenStreetMap and GeoNames) and negative examples (around 220 million) synthesized by rules. The classifier is then used to score n-gram segments of the tweet text and select the top none-overlapping candidates. We evaluate the approach on 4,500 disaster-related tweets, including around 9,500 place names from three targeted streams corresponding to the floods in Louisiana (the US), Houston (the US), and Chennai (India), respectively. We provide a comparison against several competitive baselines. The results show that the proposed approach improves the average F1-score from 0.81 for the best performing system to 0.87 (a 7\% increase).

The Architecture of ZSL-PLT is as follows:
![Screenshot](figure/workflow.jpg)

## Model Training
The first step of ZSL-PLT is to train a model based on positive examples from gazetters and negative examples sythesized by rules.
Used gazetters include OpenStreetMap and Geonames. Specifically, two boundary boxes are choosed to select the osm name entitits from OSMNames. The Geoname data in the whole US and India area is used. 
