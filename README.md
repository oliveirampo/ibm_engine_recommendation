# Recommendations with IBM Watson Studio

In this repository the following recommendation systems
were applied on a dataset with interactions that users have
with articles on the IBM Watson Studio:

* Rank-based recommendations.
* User-user based collaborative filtering.
* Matrix factorization.
* Content-based recommendations (ongoing).

The main results are presented [here]()
and the main analyses in the
[jupyter notebook](https://github.com/oliveirampo/ibm_engine_recommendation/blob/main/scr/recommendations_with_IBM.ipynb).

### File Structure

* app/ - directory with flask application for static
  [webpage](https://oliveirampo.github.io/ibm_engine_recommendation/).
* data/ - directory with input files.
* docs/ - directory with static webpage.
* scr/ - directory with scripts ans jupyter notebook.
    * recommendations_with_IBM.ipynb - jupyter notebook with analysis.
    * test.py - test for the solution.
    * top_5.p - pickle file to test top 5 rank-based recommendations.
    * top_10.p - pickle file to test top 10 rank-based recommendations.
    * top_20.p - pickle file to test top 20 rank-based recommendations.
    
### Prerequisites

Python 3.8.8

The list of packages and recommended versions are listed in file app/requirements.txt

### Installation

Install the packages listed in app/requirements.txt preferably in a virtual environment.

```python
pip install -r app/requirements.txt
```

### Credits

Udacity Data Scientist Nanodegree for providing
the input files and guidelines of the project.
[Alex Muhr](https://towardsdatascience.com/building-a-content-based-recommender-for-data-science-articles-728e5ec7d63d)
's post with an example of content-based recommender.
