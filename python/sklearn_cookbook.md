#### Pipeline

https://scikit-learn.org/stable/modules/compose.html

##### ColumnTransformer - combining transformers in series

Advantages: 

1. we can process DataFrame (and get column names after processed)
2. apply different transformer to different subsets



<https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>

<https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/>

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['pets', 'owner', 'location']
numerical_columns = ['age', 'weigth', 'height']
column_trans = make_column_transformer(
    (categorical_columns, OneHotEncoder(handle_unknown='ignore'),
    (numerical_columns, RobustScaler())
column_trans.fit_transform(df)
```

```python
from sklearn.model_selection import train_test_split
cat_attribs = ['sex','cp','fbs','restecg','exang','ca','thal']
num_attribs = ['trestbps','chol','thalach','oldpeak','slope']
X_train,X_test,y_train,y_test = train_test_split(heart_df,y,test_size=0.25,random_state=100)

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('oneHot', OneHotEncoder(categories='auto'),cat_attribs)
                       , ('minMax',MinMaxScaler(),num_attribs)])
ct_result = ct.fit_transform(X_train)
```

```python
'''
The remainder parameter can be set to an estimator to transform the remaining rating columns. By default, the remaining rating columns are ignored (remainder='drop').
We can keep the remaining rating columns by setting remainder='passthrough'. 
'''
import pandas as pd
X = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick",
               "A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3]})

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
column_trans = ColumnTransformer(
    [('city_category', OneHotEncoder(dtype='int'),['city']),
     ('title_bow', CountVectorizer(), 'title')],
    remainder='drop')

column_trans.fit(X)

column_trans.get_feature_names()

column_trans.transform(X).toarray()
```



###### with pipeline

```python
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['Salary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Age','Country']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                  ('classifier', LogisticRegression(solver='lbfgs'))])  

clf.fit(X_train,y_train)
```



##### FeatureUnion - combining transformers in parallel

Advantages:

1. apply different transformer to different subsets
2. 

Pipeline is often used in combination with [FeatureUnion](https://scikit-learn.org/stable/modules/compose.html#feature-union) which concatenates the output of transformers into a composite feature space.

A [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion) has no way of checking whether two transformers might produce identical features. It only produces a union when the feature sets are disjoint, and making sure they are the caller’s responsibility.

```python
from sklearn.model_selection import train_test_split
cat_attribs = ['sex','cp','fbs','restecg','exang','ca','thal']
num_attribs = ['trestbps','chol','thalach','oldpeak','slope']
X_train,X_test,y_train,y_test = train_test_split(heart_df,y,test_size=0.25,random_state=100)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                        ('minMax',MinMaxScaler())])
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                        ('oneHot',OneHotEncoder(categories='auto'))])
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)])
fp_result= full_pipeline.fit_transform(X_train)
```



##### Grid Search

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)

from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
```

Individual steps may also be replaced as parameters, and non-final steps may be ignored by setting them to `'passthrough'`:

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)

from sklearn.linear_model import LogisticRegression
param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
```

caching transformers: avoid repeated computation - A typical example is the case of a grid search in which the transformers can be fitted only once and reused for each configuration.

```python
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)
pipe

# Clear the cache directory when you don't need it anymore
rmtree(cachedir)
```



##### How to set parameters for transformer in pipeline ?

<https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>

The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’, as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting it to ‘passthrough’ or `None`.

```python
>>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
>>> # You can set the parameters using the names issued
>>> # For instance, fit using a k of 10 in the SelectKBest
>>> # and a parameter 'C' of the svm
>>> anova_svm.set_params(anova__k=10, svc__C=.1)
```



##### Does it support ColumnSelector or FeatureSelector ?

Solution 1:

Apart from a scalar or a single item list, the column selection can be specified as a list of multiple items, an integer array, a slice, a boolean mask, or with a [`make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector). 

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
ct = ColumnTransformer([
      ('scale', StandardScaler(),
      make_column_selector(dtype_include=np.number)),
      ('onehot',
      OneHotEncoder(),
      make_column_selector(pattern='city', dtype_include=object))])
ct.fit_transform(X)
```



Solution 2:

<https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>

```python
def all_but_first_column(X):
    return X[:, 1:]

FunctionTransformer(all_but_first_column)
```



##### Does it support modifying the sample axis (such as delete some rows) ?

<https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y>



##### [Get feature name after OneHotEncode In ColumnTransformer](https://stackoverflow.com/questions/54646709/sklearn-pipeline-get-feature-name-after-onehotencode-in-columntransformer)



##### Tips for using pipeline

1. Some transformer only have two paramters, but `Pipeline` is assuming the transformer have three parameters !

   Such as `LabelBinarizer`, `LabelEncoder`.

   Solution: [fit_transform() takes 2 positional arguments but 3 were given with LabelBinarizer](https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize) 

2. Be careful the use of transformer !

   Such as `LabelEncoder` - This transformer should be used to encode target values, *i.e.* `y`, and not the input `X`.

   <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>

   

#### Tfidf

```python
class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to CountVectorizer followed by TfidfTransformer.

	token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
     """
        def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, 	         				 lowercase=True,    
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
```

当`analyzer='word'`时,`token_pattern=r"(?u)\b\w\w+\b"` 默认匹配长度≥2的单词。

如果需要长度为1的单词，则设置`token_pattern=r"(?u)\b\w+\b"`。

```python
import re
re.findall(r'(?u)\b\w\w+\b','2.0 版本 的 中国建设银行 安全 客户端 屡次 断线 *')
Out[16]: ['版本', '中国建设银行', '安全', '客户端', '屡次', '断线']
re.findall(r'(?u)\b\w+\b','2.0 版本 的 中国建设银行 安全 客户端 屡次 断线 *')
Out[17]: ['2', '0', '版本', '的', '中国建设银行', '安全', '客户端', '屡次', '断线']
re.findall(r'\d+\.\d+|\w{1,}','2.0 版本 的 中国建设银行 安全 客户端 屡次 断线 *')
Out[23]: ['2.0', '版本', '的', '中国建设银行', '安全', '客户端', '屡次', '断线']
re.findall(r'\d+\.\d+|\*|\w{1,}','2.0 版本 的 中国建设银行 安全 客户端 屡次 断线 *')
Out[29]: ['2.0', '版本', '的', '中国建设银行', '安全', '客户端', '屡次', '断线', '*']
```



```python
class VectorizerMixin(object):
	def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)
...
class CountVectorizer(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of token counts

```





refer: https://github.com/scikit-learn/scikit-learn/issues/11834, https://stackoverflow.com/questions/29290955/token-pattern-for-n-gram-in-tfidfvectorizer-in-python





```python
class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-idf representation

    Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. This is a common term weighting scheme in information retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

    The formula that is used to compute the tf-idf of term t is
    tf-idf(d, t) = tf(t) * idf(d, t), and the idf is computed as
    idf(d, t) = log [ n / df(d, t) ] + 1 (if ``smooth_idf=False``),
    where n is the total number of documents and df(d, t) is the document frequency; the document frequency is the number of documents d that contain term t. The effect of adding "1" to the idf in the equation above is that terms with zero idf, i.e., terms  that occur in all documents in a training set, will not be entirely ignored.
    (Note that the idf formula above differs from the standard textbook notation that defines the idf as
    idf(d, t) = log [ n / (df(d, t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the
    numerator and denominator of the idf as if an extra document was seen
    containing every term in the collection exactly once, which prevents
    zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend on parameter settings that correspond to the SMART notation used in IR as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none) when ``norm=None``.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

```













