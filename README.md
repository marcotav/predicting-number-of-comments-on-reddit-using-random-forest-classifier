# Project 3

## Problem Statement

Determine which characteristics of a post on Reddit contribute most to the overall interaction as measured by number of comments.


## Preamble

In this project, we practiced some essential skills:

- Collecting data by scraping a website using the Python package `requests` and using the Python library `BeautifulSoup` which efficiently extracts HTML code. We scraped the 'hot' threads as listed on the [Reddit homepage](https://www.reddit.com/) (see figure below) and acquired the following pieces of information about each thread:

   - The title of the thread
   - subreddit that the thread corresponds to
   - The length of time it has been up on Reddit
   - The number of comments on the thread

- Using Natural Language Processing (NLP) techniques to preprocess the data. NLP, in a nutshell, is "how to transform text data and convert it to features that enable us to build models." These techniques include:

    - Tokenization (splitting text into pieces based on given patterns)
    - Removing stopwords 
    - Stemming (returns the base form of the word)
    - Lemmatization (return the word's *lemma*)

- After the step above we obtain *numerical* features which allow for algebraic computations. We then build a `RandomForestClassifier` and use it to classify each post according to the corresponding number of comments associated with it. More concretely the model predicts whether or not a given Reddit post will have above or below the _median_ number of comments.
