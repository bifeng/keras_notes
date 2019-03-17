

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













