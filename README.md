# Data-Analysis-Project

## ToDo
- How to split test and val (two seperate 0.9 embeds or one 0.8 embed)

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
- First, tune network
-- Network architecture
-- Model hyperparameters (reg. strength, dropout, batchsize)
-- Domains encoding (dimensionality, regularisation, activation function)
-- Auxiliary output in multi input model
-- Network input normalisation (coord/domains)

- Once network tuned
-- Single vs multi input network
-- Domains not coordinates and vice versa

- Probably not
-- SMOTE vs simple oversampling
-- Embedding dimensionality


## Questions
- How to do partial embedding for test and validation data
