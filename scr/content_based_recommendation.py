from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import string
import sys
import os
import re

sys.path.append("../")
from scr import IO


def clean_data(col_name, stop_list, file_name):
    """Cleans data:
    - Removes duplicates
    - Eliminates links, non alphanumerics, and punctuations.
    - Transforms all text to lowercase.

    :param col_name: (str) Name of column with text.
    :param stop_list: (arr) List of stop words.
    :param file_name: (file_name) Name of output file.
    :return: df: (dataframe) Cleaned table.
    """

    if col_name == 'title':
        df, _ = IO.read_datasets()
        df['full_title'] = df['title']
        cols_to_keep = ['article_id', 'full_title', 'title', 'email']
    else:
        _, df = IO.read_datasets()
        cols_to_keep = ['doc_description', 'doc_full_name', 'doc_status', 'article_id']

    stemmer = PorterStemmer()

    # Remove duplicate ids
    new_df = df[~df.duplicated(['article_id'])]
    new_df = new_df[cols_to_keep]

    # Ignore empty descriptions
    new_df = new_df.replace(np.nan, 'No-description', regex=True)

    new_df[col_name] = new_df[col_name].apply(clean_text)
    new_df[col_name] = new_df[col_name].apply(remove_stopwords, stop_list=stop_list)
    new_df[col_name] = new_df[col_name].apply(stem_text, stemmer=stemmer)

    # save to file
    new_df.to_csv(file_name)

    return new_df


def clean_text(text):
    """
    Eliminates links, non alphanumerics, and punctuation.
    Returns lower case text.
    :param text: (str)
    :return: (str) Cleaned text
    """

    # Remove links
    text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '', text)

    # Remove non-alphanumerics
    text = re.sub('\w*\d\w*', ' ', text)

    # Remove punctuation and apply lowercase
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())

    # Remove newline characters
    text = text.replace('\n', ' ')

    return text


def remove_stopwords(text, stop_list):
    """
    Removes stop words.

    :param text: (str)
    :param stop_list: (arr) List of stop words.
    :return: text (str) Cleaned text.
    """

    clean_text_lst = []
    for word in text.split(' '):
        if (word not in stop_list) and (len(word) > 2):
            clean_text_lst.append(word)
    return ' '.join(clean_text_lst)


def stem_text(text, stemmer):
    """
    Stems words.
    :param text: (str)
    :param stemmer: (Stemmer object)
    :return: text (str) Cleaned text.
    """

    word_list = []
    for word in text.split(' '):
        word_list.append(stemmer.stem(word))
    return ' '.join(word_list)


def apply_TFIDF_and_SVD(df, col_name, stop_list, num_topics):
    """
    Applies TFIDF and SVD to document description.
    :param df: (pandas dataframe)
    :param col_name: (str) Name of column with text.
    :param stop_list: (arr) List of stop words.
    :param num_topics: (int) Number of topics.
    :return:
    """

    vectorizer = TfidfVectorizer(stop_words=stop_list, ngram_range=(1, 1))
    doc_word = vectorizer.fit_transform(df[col_name])

    svd = TruncatedSVD(num_topics)
    docs_svd = svd.fit_transform(doc_word)

    return svd, vectorizer


def apply_NMF(df, col_name, stop_list, num_topics):
    """
    Applies TFIDF and NMF to document description.
    :param df: (pandas dataframe)
    :param col_name: (str) Name of column with text.
    :param stop_list: (arr) List of stop words.
    :param num_topics: (int) Number of topics.
    :return:
    """

    vectorizer = TfidfVectorizer(stop_words=stop_list, ngram_range=(1, 1))
    doc_word = vectorizer.fit_transform(df[col_name])

    nmf = NMF(num_topics, init='nndsvda')
    docs_nmf = nmf.fit_transform(doc_word)

    return docs_nmf, nmf, vectorizer


def display_topics(model, feature_names, num_top_words, num_top_topics, topic_names=None):
    """
    Displays topic and key words.
    :param model: (svd model)
    :param feature_names: (arr) Features of vectorizer object
    :param num_top_words: (int) Number of words.
    :param num_top_topics: (int) Number of topics.
    :param topic_names: (str) Name of topics.
    :return:
    """

    count = 0
    for idx, topic in enumerate(model.components_):
        if count == num_top_topics:
            break
        if (not topic_names) or (not topic_names[idx]):
            print("\nTopic ", (idx + 1))
        else:
            print("\nTopic: '", topic_names[idx], "'")

        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-num_top_words - 1:-1]]))
        count += 1


def save_topics(col_name, num_topics, docs, df):
    """

    :param num_topics: (int)
    :param docs: (np.ndarray)
    :param df: (pandas dataframe)
    :return:
        doc_topic_df: (pandas dataframe)
    """

    # Create topic sum for each article. Later remove all articles with sum 0.
    topic_sum = pd.DataFrame(np.sum(docs, axis=1))

    # Turn our docs array into a data frame
    doc_topic_df = pd.DataFrame(data=docs)

    # Define column names for dataframe
    # Merge all of our article metadata and name columns
    if col_name == 'title':
        column_names = ['article_id', 'full_title', 'title', 'email']
        for i in range(num_topics):
            column_names.append('Topic-{}'.format(i))
        column_names.append('sum')

        doc_topic_df = pd.concat([df[['article_id', 'full_title', 'title', 'email']], doc_topic_df, topic_sum], axis=1)
        doc_topic_df.columns = column_names
    else:
        column_names = ['doc_full_name', 'doc_status', 'article_id']
        for i in range(num_topics):
            column_names.append('Topic-{}'.format(i))
        column_names.append('sum')

        doc_topic_df = pd.concat([df[['doc_full_name', 'doc_status', 'article_id']], doc_topic_df, topic_sum], axis=1)
        doc_topic_df.columns = column_names

    # Remove articles with topic sum = 0, then drop sum column
    doc_topic_df = doc_topic_df[doc_topic_df['sum'] != 0]
    doc_topic_df.drop(columns='sum', inplace=True)

    # Reset index then save
    doc_topic_df.reset_index(drop=True, inplace=True)

    return doc_topic_df


def compute_dists(top_vec, topic_array, norms):
    """
    Returns cosine distances for top_vec compared to every article
    :param top_vec: (np.array)
    :param topic_array: (np.array)
    :param norms: (np.array)
    :return:
    """

    dots = np.matmul(topic_array, top_vec)
    input_norm = np.linalg.norm(top_vec)
    co_dists = dots / (input_norm * norms)
    return co_dists


def produce_rec(num_articles, num_topics, top_vec, topic_array, norms, doc_topic_df, rand=15):
    """
    Produces a recommendation based on cosine distance.
    Rand variable controls level of randomness in output recommendation.

    :param num_articles: (int) Number of articles to recommend
    :param num_topics: (int) Number of topics
    :param top_vec: (np.array)
    :param topic_array: (np.array)
    :param norms: (np.array)
    :param doc_topic_df: (pandas dataframe)
    :param rand: (int)
    :return:
        rec: (pandas series)
    """
    # Add a bit of randomness to top_vec
    # top_vec = top_vec + np.random.rand(num_topics,)/(np.linalg.norm(top_vec)) * rand
    top_vec = top_vec / (np.linalg.norm(top_vec))
    co_dists = compute_dists(top_vec, topic_array, norms)

    # get article with highest similarity
    # rec = doc_topic_df.loc[np.argmax(co_dists)]

    # get first N articles with highest similarity
    N = num_articles + 1
    idxs = co_dists.argsort()[-N:][::-1]
    # ignore first entry (the same article)
    rec = doc_topic_df.loc[idxs[1:]]

    return rec


def get_similar_articles(col_name, article_pos, num_articles, num_topics, doc_topic_df):
    topic_names = ['Topic-{}'.format(i) for i in range(num_topics)]
    topic_array = np.array(doc_topic_df[topic_names])
    norms = np.linalg.norm(topic_array, axis=1)

    top_vec = topic_array[article_pos]
    rec = produce_rec(num_articles, num_topics, top_vec, topic_array, norms, doc_topic_df)

    if col_name == 'title':
        print(doc_topic_df.iloc[article_pos][['article_id', 'full_title']])
        print(doc_topic_df.iloc[article_pos][topic_names])
        print(rec[['full_title', 'article_id']])
    else:
        print(doc_topic_df.iloc[article_pos][['doc_full_name', 'article_id']])
        print(rec[['doc_full_name', 'article_id']])


def recommend():
    stop_list = STOPWORDS.union(set(['data', 'ai', 'learning', 'time', 'machine', 'like', 'use',
                                     'new', 'intelligence', 'need', "it's", 'way', 'artificial',
                                     'based', 'want', 'know', 'learn', "don't", 'things', 'lot',
                                     "let", 'model', 'input', 'output', 'train', 'training',
                                     'trained', 'it', 'we', 'you']))

    file_name = '../data/articles_cleaned.csv'
    col_name = 'doc_description'
    # col_name = 'title'

    df = clean_data(col_name, stop_list, file_name)

    num_topics = 15
    num_words = 10
    # docs, svd, vectorizer = apply_TFIDF_and_SVD(df, col_name, stop_list, num_topics)
    docs, svd, vectorizer = apply_NMF(df, col_name, stop_list, num_topics)
    # display_topics(svd, vectorizer.get_feature_names(), num_words, num_topics)

    doc_topic_df = save_topics(col_name, num_topics, docs, df)

    if col_name == 'title':
        article_idx = df[df['article_id'] == 1427.00].index[0]
        get_similar_articles(col_name, article_idx, 5, num_topics, doc_topic_df)

    else:
        article_idx = 1052
        get_similar_articles(col_name, article_idx, 5, num_topics, doc_topic_df)

    # print(df['doc_full_name'])


def main():
    recommend()


if __name__ == '__main__':
    main()
