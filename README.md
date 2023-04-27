# Hashing pruned embeddings with a fruit-fly inspired algorithm

## Table of Contents

- [Description](#Experiment Description)
- [Data](#Data)
- [Setup](#Experimental Setup)
- [Results](#Results)
- [Usage](#Usage)
- [References](#References)

## Experiment Description

This experiment aims to investigate the correlation between pruned and unpruned GloVe embeddings and human similarity judgments. The hypothesis is that the pruned embeddings will have a higher correlation than the unpruned embeddings.

The experiment uses an algorithm inspired by the fruit fly's olfactory system to hash pruned GloVe embeddings and their unpruned counterparts. The hashing algorithm parameters that are tuned include the number of Kenyon cells, the number of projection neurons, the winner-takes-all layer, and the number of principal components.

## Data

The experiment uses two datasets:

    Human similarity judgments dataset collected by Richie & Bhatia (2020) that contains similarity data for words in eight categories (https://osf.io/d7fm2/?view_only=c5ba5d34a5e34ff3970a652c07aadc5c.), and
    MEN similarity dataset (Bruni et al. 2014) that contains 3000 pairs of similarity judgments crowdsourced by native speakers of English.

The GloVe embeddings used for both datasets were obtained by training on the GigaWord Corpus and Wikipedia, yielding vectorial representation for each word with a dimensionality of 300.
Model

The algorithm used for similarity hashing is inspired by a fruit fly's olfactory system described by Dasgupta et al. (2017) and was adapted from_ https://github.com/ml-for-nlp/fruit-fly.

## Experimental Setup

The experimental setup involves pruning the GloVe embeddings and then hashing the pruned and unpruned embeddings.

For pruning the embeddings, the features of the embedding matrix are ranked based on how much the Spearman's correlation between the embeddings matrix and the human similarity matrix increases or decreases after the removal of the feature. The set of features that produces the highest 2OI (the amount of information retained after pruning) are chosen by the pruning algorithm.

After pruning, the pruned and unpruned embeddings are hashed using the fruit fly-inspired algorithm. The correlation between the distance of the hashed results and the MEN similarity embedding is calculated.

The experiment tests different scenarios with varying numbers of features, projection neurons, Kenyon cells, and principal components.


## Results

The experiments conducted did not support the initial hypothesis that the hashed version of the pruned embeddings would perform better in terms of correlations to the MEN dataset similarity pairs. Interestingly, it was observed that there was no single feature retained across all categories after the pruning process, which suggests that pruning solutions might be category-specific. The difference between hashed and unhashed embeddings was found to be marginal, which could be attributed to the randomness of the projections. It was suggested that running the experiments several times and obtaining mean and standard deviation scores could lead to more precise results. The findings indicate that using a larger corpus that can be divided into training and testing sets may yield different results, and further exploration of the resulting semantic spaces is warranted.

## Usage

This is the accompanying code for an experiment for hashing pruned and unpruned GloVe word embeddings using a fruit-fly-derived hashing algorithm. It is based on the repository https://github.com/ml-for-nlp/fruit-fly. To run it, use :

    python3 pruned_fly_solution_pca.py --matrix_type=pruned --num_cat=2

## References

- Bruni, E., Tran, N.K., & Baroni, M. (2014). Multimodal Distributional Semantics. J. Artif. Intell. Res., 49, 1-47.

- Dasgupta, S., Stevens, C.F., & Navlakha, S. (2017). A neural algorithm for a fundamental computing problem. Science, 358, 793 - 796.

- Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation. Conference on Empirical Methods in Natural Language Processing.

- Richie, R., & Bhatia, S. (2020). Similarity Judgment Within and Across Categories: A Comprehensive Model Comparison. Cognitive science, 45 8, e13030 .
