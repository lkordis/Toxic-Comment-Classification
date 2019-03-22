# Multi-label Toxic Language Classification of Wikipedia Comments
## Lovro Kordiš, Marko Šmitran

The paper tackles the problem of classification of toxic text in Wikipedia comments featured in Toxic Comment Classification Chal-
lenge. It is a multi-label problem in nature as each comment can fall into multiple categories of toxicity. We approach this task from
both classical machine learning and deep-learning angles. In our classical ML approach we used several classifiers, one for binary clas-
sification of each category of toxicity. Classifiers were trained on already NLP pre-processed data. We also tried different categories of
models – Linear SVM and Logistic Regression. For our deep-learning approach we used models based on LSTMs. Evaluation of models
was done on data provided in the challenge. In addition to evaluating models on provided labeled data we further evaluated them with
Challenge’s official automatic evaluation.
