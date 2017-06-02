# Data-Analysis-Project

## ToDo
- Use embedding layer for training
- Include automatic NN metric logging
- Pass val. data directly (not with generator)

- Try training on just domains

- Add more data:
-- Node degree (log transform)
-- Clusering coeff (nodewise)
-- Betw. centrality/ and others

- Use crossvalidation to validate NN sklearn.cross_validation.StratifiedKfold
- Find way to replace embedding

- Website
-- viz: different network structures and their performance - interactive
-- viz: subsample embedding

## Comparisons
- Model hyperparameters (no. of layers/units, reg. strength, dropout, batchsize)
- SMOTE vs simple oversampling
- Single vs multi input network
- Embedding dimensionality
- Domains encoding (dimensionality, regularisation, activation function)
- Auxilary output in multi input model
- Domains not coordinates and vice versa


## Questions
- How to do partial embedding for test and validation data
- Is it actually okay to train the autoencoder without holdout?
- What other data to include
