from flask_bootstrap import Bootstrap
from flask_flatpages import FlatPages
from flask import render_template
from flask_frozen import Freezer
from flask import Flask
import plotly
import json
import sys

sys.path.append("../")
from scr import IO
from scr import plot
from scr import recommendation

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FREEZER_RELATIVE_URLS = True
FREEZER_IGNORE_404_NOT_FOUND = True
FREEZER_DESTINATION='../docs'

app = Flask(__name__)
app.config.from_object(__name__)
Bootstrap(app)

pages = FlatPages(app)
freezer = Freezer(app)

@app.route('/')
def page_index():
    # data sets
    df, df_content = IO.read_datasets()

    # distribution of articles per user
    interactions_count = df.groupby('email')['article_id'].count().values
    # title = 'Distribution of articles per user'
    title = ''
    x_label = 'Articles per user'
    histogram_articles_user = plot.plot_histogram(interactions_count, x_label, '', title)

    histogram_articles_user_JSON = json.dumps(histogram_articles_user, cls=plotly.utils.PlotlyJSONEncoder)

    top_10_articles = recommendation.get_top_articles(10, df)

    file_name = '../data/top_10_articles_by_topic.json'
    with open(file_name, 'r') as json_file:
        top_10_articles_by_topic = json.load(json_file)

    email_encoded = IO.email_mapper(df)
    del df['email']
    df['user_id'] = email_encoded
    user_item = IO.create_user_item_matrix(df)
    user_item = user_item.iloc[:, 0:15]

    file_name = '../data/similar_users.json'
    with open(file_name, 'r') as json_file:
        similar_users = json.load(json_file)

    return render_template('index.html', df=df, df_content=df_content,
                           histogram_articles_user=histogram_articles_user_JSON,
                           top_10_articles=top_10_articles,
                           top_10_articles_by_topic=top_10_articles_by_topic,
                           user_item=user_item, similar_users=similar_users)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # app.run(debug=True)
        freezer.freeze()
    else:
        app.run(debug=True)