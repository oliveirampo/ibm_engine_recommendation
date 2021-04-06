import pandas as pd
import numpy as np
import sys

sys.path.append("../")
from scr import IO


def get_top_articles(n, df):
    """
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles

    """
    idx = df.article_id.value_counts().sort_values(ascending=False).head(n).index
    top_articles = df[df['article_id'].isin(idx)]['title'].unique()

    return top_articles  # Return the top article titles from df (not df_content)


def get_top_10_by_topic():
    df, _ = IO.read_datasets()
    df = df[['article_id', 'title']]

    data = {"name": "articles", "children": []}

    keywords = [{'parent': 'data', 'children': ['analyze', 'big data', 'visualize']},
                {'parent': 'database', 'children': []},
                {'parent': 'data science', 'children': []},
                {'parent': 'machine learning', 'children': ['algorithm', 'api', 'classif',
                                                            'predict', 'python', 'recommend',]},
                {'parent': 'libraries', 'children': ['keras', 'pandas', 'regression', 'scikit', 'tensorflow']},
                {'parent': 'model', 'children': ['bayes', 'neural', 'regression']}]

    for item in keywords:
        parent_key = item['parent']
        # print(parent_key)

        parent_data = {"name": parent_key, "children": []}

        if parent_key == 'libraries':
            for child_key in item['children']:
                # print('\t', child_key)
                top_10 = get_top_10_helper(df, child_key)

                if len(top_10) != 0:
                    child_data = {"name": child_key, "children": top_10}
                    parent_data["children"].append(child_data)

            data["children"].append(parent_data)

        elif parent_key in ['database', 'data science']:
            top_10 = get_top_10_helper(df, parent_key)
            parent_data['children'] = top_10
            data["children"].append(parent_data)

        else:
            df_parent = df[df['title'].str.contains(parent_key)]

            for child_key in item['children']:
                # print('\t', child_key)
                top_10 = get_top_10_helper(df_parent, child_key)

                if len(top_10) != 0:
                    child_data = {"name": child_key, "children": top_10}
                    parent_data["children"].append(child_data)

            data["children"].append(parent_data)

    return data


def get_top_10_helper(df, child_key):
    """Get top 10 articles with capitalized title."""

    df_child = df[df['title'].str.contains(child_key)]

    top_10 = get_top_articles(10, df_child)
    top_10 = np.array(top_10, dtype=str)
    top_10 = np.char.capitalize(top_10)

    # top_10 = top_10.tolist()
    data = []
    for idx, article in enumerate(top_10):
        if '"' in article:
            print(article)
        if "i ranked every" in article:
            print(article)
            continue

        article = article.replace('"', '\'')

        data.append({"name": article, "value": idx})

    return data


def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook

    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the title column)
    '''
    # Your code here
    article_id = map(float, article_ids)

    article_names = df[df['article_id'].isin(article_id)]['title'].drop_duplicates().values.tolist()

    return article_names  # Return the article names associated with list of article ids


def get_user_articles(df, user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise

    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids
                    (this is identified by the doc_full_name column in df_content)

    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # print(user_item)
    # print(user_id)
    # user_item.loc[user_id]
    # sys.exit('STOP')
    article_ids = (user_item.loc[user_id][user_item.loc[user_id].values == 1].value.index).to_list()
    article_ids = ['{}'.format(i) for i in article_ids]

    article_names = get_article_names(article_ids, df)

    return article_ids, article_names  # return the ids and names


def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    user_item - (pandas dataframe) matrix of users by articles:
            1's when a user has interacted with an article, 0 otherwise


    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u

    Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                    highest of each is higher in the dataframe

    '''
    dot_prod_articles = user_item.dot(np.transpose(user_item))
    sim = dot_prod_articles.loc[user_id].sort_values(ascending=False)
    sim = sim.to_frame().reset_index().drop(0, axis=0)
    sim.columns = ['user_id', 'similarity']

    neighbor_counts = pd.Series(df['user_id'].value_counts())
    neighbor_counts = neighbor_counts.to_frame().reset_index()
    neighbor_counts.columns = ['user_id', 'user_interactions']

    neighbors_df = pd.merge(sim, neighbor_counts[['user_id', 'user_interactions']], on='user_id', how='left')
    neighbors_df.columns = ['user_id', 'similarity', 'user_interactions']
    neighbors_df = neighbors_df.sort_values(['similarity', 'user_interactions'], ascending=False).reset_index()

    neighbors_df = neighbors_df.drop('index', axis=1)

    return neighbors_df  # Return the dataframe specified in the doc_string


def get_all_similar_users():
    """
    OUTPUT:
    data - json file

    Description:
    For a list of users get their most similar users.
    """

    df, _ = IO.read_datasets()

    email_encoded = IO.email_mapper(df)
    del df['email']
    df['user_id'] = email_encoded
    user_item = IO.create_user_item_matrix(df)

    data = {"name": "users"}
    children = []
    n_users = 101
    user_considered = [i for i in range(1, n_users)]

    for id in user_considered:
        top_users = get_top_sorted_users(id, df, user_item)
        top_users = top_users.loc[0:10, 'user_id'].tolist()

        intersection = np.intersect1d(top_users, user_considered)
        intersection = ['users.{}'.format(id) for id in intersection]

        child = {"name": id, "size": 1, "imports": intersection}
        children.append(child)

    data["children"] = children

    return data
