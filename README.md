<img align="right" width="120" height="120"
     title="Size Limit logo" src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/Reddit-logo.png">

# Predicting Comment on Reddit 

<br>

## Problem Statement

Determine which characteristics of a post on Reddit contribute most to the overall interaction as measured by number of comments.

## Preamble

In this project, we practiced some essential skills:

- Collecting data by scraping a website using the Python package `requests` and using the Python library `BeautifulSoup` which efficiently extracts HTML code. We scraped the 'hot' threads as listed on the [Reddit homepage](https://www.reddit.com/) (see figure below) and acquired the following pieces of information about each thread:

   - The title of the thread
   - subreddit that the thread corresponds to
   - The length of time it has been up on Reddit
   - The number of comments on the thread
  
  <br>
 
   <img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditpage.png" width="800">
   
- Using Natural Language Processing (NLP) techniques to preprocess the data. NLP, in a nutshell, is "how to transform text data and convert it to features that enable us to build models." These techniques include:

   - Tokenization (splitting text into pieces based on given patterns)
   - Removing stopwords 
   - Stemming (returns the base form of the word)
   - Lemmatization (return the word's *lemma*)
   
- After the step above we obtain *numerical* features which allow for algebraic computations. We then build a `RandomForestClassifier` and use it to classify each post according to the corresponding number of comments associated with it. More concretely the model predicts whether or not a given Reddit post will have above or below the _median_ number of comments.
    
### Writing functions to extract the items above

The functions below will extract the information we need:

```
def extract_title_from_result(result,num=25):
    titles = []
    title = result.find_all('a', {'data-event-action':'title'})
    for i in title:
        titles.append(i.text)
    return titles

def extract_time_from_result(result,num=25):
    times = []
    time = result.find_all('time', {'class':'live-timestamp'})
    for i in time:
        times.append(i.text)
    return times

def extract_subreddit_from_result(result,num=25):
    subreddits = []
    subreddit = result.find_all('a', {'class':'subreddit hover may-blank'})
    for i in subreddit:
        subreddits.append(i.string)
    return subreddits

def extract_num_from_result(result,num=25):
    nums_lst = []
    nums = result.find_all('a', {'data-event-action': 'comments'})
    for i in nums:
        nums_lst.append(i.string)
    return nums_lst
```

 We then write a function that finds the last `id` on the page, and stores it:
 
 ```
def get_urls(n=25):
    j=0   # counting loops
    titles = []
    times = []
    subreddits = []
    nums = []
    URLS = []
    URL = "http://www.reddit.com"
    
    for _ in range(n):
        
        res = requests.get(URL, headers={"user-agent":'mt'})
        soup = BeautifulSoup(res.content,"lxml")
        
        titles.extend(extract_title_from_result(soup))
        times.extend(extract_time_from_result(soup))
        subreddits.extend(extract_subreddit_from_result(soup))
        nums.extend(extract_num_from_result(soup))         

        URL = soup.find('span',{'class':'next-button'}).find('a')['href']
        URLS.append(URL)
        j+=1
        print(j)
        time.sleep(3)
        
    return titles, times, subreddits, nums, URLS
 ```

We then build a `DataFrame`, perform some EDA and create:

- a binary column that classifies the number of comments
comparing the values with their median
- A set of dummy columns for the subreddits
- Concatenate both

```
df['binary'] = df['nums'].apply(lambda x: 1 if x >= np.median(df['nums']) else 0)
df_subred = pd.concat([df['binary'],pd.get_dummies(df['subreddits'], drop_first = True)], axis = 1)
```

To preprocess the text before creating numerical features from the text (see below) we build the following `cleaner` function:

```
def cleaner(text):
    stemmer = PorterStemmer()                                          
    stop = stopwords.words('english')    
    text = text.translate(str.maketrans('', '', string.punctuation))   
    text = text.translate(str.maketrans('', '', string.digits))        
    text = text.lower().strip() 
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)
```

I then use `CountVectorizer` to create features based on the words in the thread titles. We will then combine this new table `df_all` and the subreddits features table and build a new model.

```
cvt = CountVectorizer(min_df=min_df, preprocessor=cleaner)
cvt.fit(df["titles"])
cvt.transform(df['titles']).todense()
X_title = cvt.fit_transform(df["titles"])
X_thread = pd.DataFrame(X_title.todense(), 
                        columns=cvt.get_feature_names())
df_all = pd.concat([df_subred,X_thread],axis=1)                     
```

<img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditwordshist.png" width="400">



Finally, now with the data properly treated, we use the following function to fit the training data using a `RandomForestClassifier` with optimized hyperparameters obtained using `GridSearchCV`:

```
n_estimators = list(range(20,220,10))
max_depth = list(range(2, 22, 2)) + [None]

def rfscore(df,target_col,test_size,n_estimators,max_depth):
    
    X = df.drop(target_col, axis=1)   # predictors
    y = df[target_col]                # target
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, test_size = test_size, random_state=42) # TT split
    rf_params = {
             'n_estimators':n_estimators,
             'max_depth':max_depth}   # parameters for grid search
    rf_gs = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, verbose=1, n_jobs=-1)
    rf_gs.fit(X_train,y_train) # training the random forest with all possible parameters
    max_depth_best = rf_gs.best_params_['max_depth']      
    n_estimators_best = rf_gs.best_params_['n_estimators'] 
    best_rf_gs = RandomForestClassifier(max_depth=max_depth_best,n_estimators=n_estimators_best) 
    best_rf_gs.fit(X_train,y_train)  
    best_rf_score = best_rf_gs.score(X_test,y_test) 
    preds = best_rf_gs.predict(X_test)
    feature_importances = pd.Series(best_rf_gs.feature_importances_, index=X.columns).sort_values().tail(5)
    print(feature_importances.plot(kind="barh", figsize=(6,6)))
    return 
```
We then use the function below that performs cross-validation, to obtain our accuracy score (using the model with best parameters obtained from the `GridSearch`):

```
def cv_score(X,y,cv,n_estimators,max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators_best,
                                max_depth=max_depth_best)
    s = cross_val_score(rf, X, y, cv=cv, n_jobs=-1)
    return("{} Score is :{:0.3} ± {:0.3}".format("Random Forest", s.mean().round(3), s.std().round(3)))
```

<br>

   <img src="https://github.com/marcotav/predicting-the-number-of-comments-on-reddit/blob/master/redditRF.png" width="400">



