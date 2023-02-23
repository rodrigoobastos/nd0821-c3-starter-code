# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Developer: Rodrigo da Matta Bastos

Model date: 07/02/2023

Model version: v1.0.0

Model type: Random Forest

References: https://scikit-learn.org/stable/modules/ensemble.html#forest

## Intended Use

This model's objective is to predict whether income exceeds $50k/yr on the Census Income (AKA Adult) Data Set.

This model was developed for educational purposes and shouldn't be used as an oficial solution for the Adult Data Set. 

## Training Data
Census Income Data Set (80% of the data)

## Evaluation Data
Census Income Data Set (20% of the data)

## Metrics
This model was evaluated with precision, recall, and F1 Score. The scores are shown bellow:

Precision: 0.781
Recall: 0.604
F1: 0.681

This model was also evaluated on education slices. The table bellow shows performance for each education level:

Education Level | Preschool | 1st-4th | 5th-6th | 7th-8th | 9th | 10th | 11th | 12th | HS-grad | Some-college | Assoc-voc | Assoc-acdm | Bachelors | Masters | Prof-school | Doctorate
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
Precision | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 0.71 | 0.70 | 0.75 | 0.73 | 0.80 | 0.84 | 0.75
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
Recall | 1.00 | 0.00 | 0.29 | 0.33 | 0.11 | 0.08 | 0.67 | 0.17 | 0.28 | 0.44 | 0.43 | 0.56 | 0.82 | 0.84 | 0.83 | 0.87
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
F1 | 1.00 | 0.00 | 0.44 | 0.40 | 0.20 | 0.14 | 0.80 | 0.29 | 0.42 | 0.55 | 0.54 | 0.64 | 0.78 | 0.82 | 0.84 | 0.80

## Ethical Considerations
Data is based on a public dataset of 1994 Census.

## Caveats and Recommendations
Model was trained with 1994 data and can be unsuited for use on recent data
Model have higher performance on more educated individuals and shouldn't be used in a real life scenario due to potential bias.