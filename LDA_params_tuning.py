import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gensim, spacy, re

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parameters_tuning(corpus, dictionary):
    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 5
    max_topics = 30
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 'symmetric', 'asymmetric']

    # Beta parameter
    beta = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 'symmetric']

    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.25)),
                   gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.5)),
                   gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
                   corpus]

    # Coherence type
    coherence = ['c_npmi', 'c_uci', 'u_mass', 'c_v']

    corpus_title = ['25% Corpus', '50% Corpus', '75% Corpus', '100% Corpus']

    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }

    total_iterations = len(coherence) * len(alpha) * len(beta) * len(corpus_title) * len(topics_range)

    # Can take a long time to run
    pbar = tqdm.tqdm(total=total_iterations)
    # iterate through validation corpora
    for i in range(len(corpus_sets)):
        for c in coherence:
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterate through beta values
                    for b in beta:
                        #  get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i],
                                                      dictionary=dictionary,
                                                      k=k,
                                                      a=a,
                                                      b=b,
                                                      coherence=c)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        # model_results['Measure'].append(c)
                        # print(model_results['Coherence'])

                        pbar.update(1)

    pd.DataFrame(model_results).to_csv('./lda_tuning_results.csv', index=False)
    pbar.close()


def compute_coherence_values(corpus, dictionary, k, a, b, coherence):
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=k,
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       alpha=a,
                                       eta=b)

    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=processed_docs,
                                         dictionary=id2word,
                                         coherence=coherence)

    return coherence_model_lda.get_coherence_per_topic()


def make_corpus(docs, dictionary):
    """ For each document in the corpus, we map the words to their ids
        and see how many times they appear in the document
        (id/word, n_appearances_in_document)"""
    created_corpus = [dictionary.doc2bow(text, allow_update=True) for text in docs]
    return created_corpus


def make_ngrams(texts, ngrams):
    ngram_model = gensim.models.phrases.Phraser(ngrams)
    return [ngram_model[doc] for doc in texts]


def phrases(docs):
    """ Function to create gensim Phraser
    for bigrams and trigrams """
    return gensim.models.Phrases(docs, threshold=5)


def preprocessing(text):
    """ Preprocess text: remove words
     that have less than 3 tokens """
    result = []
    for token in simple_preprocess(text):
        if len(token) >= 3:
            result.append(token)
    return result


if __name__ == "__main__":
    # Takes in input a .tsv with lemmatized texts
    # .tsv structure is: post_id\tlemmatized_text\n
    lemmatized_path = './lemmatized_text.tsv'

    # Read data
    df = pd.read_csv(lemmatized_path, sep='\t', encoding='utf-8', header=None,
                     lineterminator='\n', names=['post_id', 'body_lemmat'], nrows=100)
    # Drop lines with empty 'body_lemmat'
    df.dropna(inplace=True)

    processed_docs = df['body_lemmat'].map(preprocessing)
    print(len(processed_docs))

    # Make bigrams and trigrams
    bigrams = phrases(processed_docs)
    trigrams = phrases(bigrams[processed_docs])
    words_bigrams = make_ngrams(processed_docs, bigrams)
    words_trigrams = make_ngrams(words_bigrams, trigrams)

    # Dictionary containing mapping between normalized words and integer ids
    id2word = gensim.corpora.Dictionary(words_trigrams)

    # Filter id2word Dictionary and remove words that appear in less than 100 posts or in more than 70% of documents
    id2word.filter_extremes(no_below=100, no_above=0.7)

    corpus = make_corpus(processed_docs, id2word)

    parameters_tuning(corpus, id2word)
