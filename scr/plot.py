from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import Image
import numpy as np


# import IO


def plot_histogram(data, x_label, y_label, title):
    """Plot histogram from array.

    :param data: (ndarray) Data.
    :param x_label: (str) X-axis label.
    :param y_label: (str) Y-axis label.
    :param title: (str) Plot title.
    :return:
        fig (figure object)
    """

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        marker_color='#2f4f4f'
    ))

    fig.update_layout(
        title_text=title,
        plot_bgcolor="#FFF",  # Sets background color to white
        xaxis=dict(
            title=x_label,
            range=[0, 50],
            linecolor="#BCCCDC",  # Sets color of X-axis line
            showgrid=False  # Removes X-axis grid lines
        ),
        yaxis=dict(
            title=y_label,
            linecolor="#BCCCDC",  # Sets color of Y-axis line
            showgrid=False,  # Removes Y-axis grid lines
        )
    )

    return fig


def create_word_cloud_from_titles():
    mask = np.array(Image.open("../app/static/images/square.png"))

    df, _ = IO.read_datasets()

    # word could with title of articles
    titles = " ".join(txt for txt in df['title'])

    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color="#dcdcdc", mode="RGBA", max_words=100, mask=mask).generate(titles)

    # create coloring from image
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[7, 7])
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")

    # store to file
    plt.savefig("../app/static/images/word_cloud_titles.png", format="png")
    # wordcloud.to_file("../app/static/images/word_cloud_titles.png")

    plt.show()


def main():
    # create_word_cloud_from_titles()
    print('Nothing to do.')


if __name__ == '__main__':
    main()
