"""This module provides functions to handle input and output files.

Functions:
    read_datasets
"""

import pandas as pd
import numpy as np
import json
import sys

sys.path.append("../")
from scr import recommendation


def read_datasets():
    """Read csv input files into data frames

    :return:
        df: (pandas dataframe) Maps users to articles.
        df_content: (pandas dataframe) Contains information about articles.
    """

    df = pd.read_csv('../data/user-item-interactions.csv')
    df_content = pd.read_csv('../data/articles_community.csv')
    del df['Unnamed: 0']
    del df_content['Unnamed: 0']

    return df, df_content


def write_json_file(data, file_name):
    """Write dictionary to JSON file"""

    with open(file_name, 'w') as file:
        json.dump(data, file, indent=1)


def email_mapper(df):
    coded_dict = dict()
    cter = 1
    email_encoded = []

    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        email_encoded.append(coded_dict[val])
    return email_encoded


def create_user_item_matrix(df):
    """
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
    an article and a 0 otherwise
    """

    df = df[['user_id', 'article_id']]
    df['value'] = 1
    df = df.groupby(['user_id', 'article_id']).max().unstack()
    df = df.fillna(0)
    user_item = df

    return user_item  # return the user_item matrix


def main():
    # top_10_by_topic = recommendation.get_top_10_by_topic()
    # print(top_10_by_topic['children'][1])
    # file_name = '../data/top_10_articles_by_topic.json'
    # write_json_file(top_10_by_topic)

    file_name = '../data/similar_users.json'
    data = recommendation.get_all_similar_users()
    write_json_file(data, file_name)


if __name__ == '__main__':
    main()
