# Protonet
This Repository contains the code for the data analysis project completed as part of the MSc in Bioinformatics and Theoretical Systems Biology at Imperial College London 2016/2017.

## Lay Summary
Artificial neural networks are a type of machine learning technique that can enable computers to learn complex patterns and relationships from data. This technique has been successfully employed in the fields of speech recognition, natural language understanding and computer vision. One prominent example of their application are personal assistants like Apple Siri.

In this project we have applied an artificial neural network to a biological problem: predicting novel protein-protein interactions (PPIs) in the human body by integrating a diverse range of input data sources. PPIs form the basis of many biological processes such as transport across membranes, cell signalling and the immune response. Discovering novel interactions can therefore help us find new drug targets and therapies, but can also improve our understanding of protein function and the basis of their interaction.

We combined three separate data sources to train the neural network. Firstly, the coordinates of each protein after embedding them in a multidimensional space (interacting proteins being placed closer together). Secondly, information about the known domains of each protein. Lastly, the gene ontology annotation for each protein.

When artificially removing interactions from the network, our model was able to recover 28 % of them with 13 % of all predicted interactions being correct. This result demonstrates the potential of applying novel machine learning techniques to biological problems.

## Abstract
Current biochemical high-throughput methods for detecting novel protein-protein interactions are prone to significant Type I and Type II errors. Computational tools with the ability to produce high-quality predictions have hence been investigated for several years.

In this work, we present a feed-forward artificial neural network for the prediction of novel protein-protein interactions. We combine available interaction data, Gene Ontology annotations and protein domain data as inputs to our model. The nature of our model provides fully automated data integration without requiring manual weighting of the different input data sources.

We were able to reconstruct artificially removed edges from the human interactome with a Matthews correlation coefficient of 0.19 and F1 score of 0.17. Considering the complexity of the problem,  this result highlights the potential of combining supervised and unsupervised machine learning approaches to solve biological problems using a diverse range of input data types.
