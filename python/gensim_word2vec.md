

```
w2v_model = KeyedVectors.load_word2vec_format(w2v_model, binary=False, limit=50000)
w2v_model.vectors
w2v_model.vectors.data
w2v_model.vocab  # Each word in the vocabulary has an associated vocabulary object, which contains an index and a count.
...
```



### `syn0`, `syn0norm`,`syn1`,`syn1neg`

These names were inherited from the original Google `word2vec.c` implementation.

`syn0` array essentially holds raw word-vectors. 

`syn0norm` array is filled with these unit-normalized vectors (This makes the cosine-similarity calculation easier.) 

`syn1` (or `syn1neg` in the more common case of negative-sampling training) properties, when they exist on a full model (and not for a plain `KeyedVectors` object of only word-vectors), are the model neural network's internal 'hidden' weights leading to the output nodes. They're needed during model training, but not a part of the typical word-vectors collected after training.

https://stackoverflow.com/questions/53301916/python-gensim-what-is-the-meaning-of-syn0-and-syn0norm




### init_sims

```python
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.vectors.shape[0]):
                    self.vectors[i, :] /= sqrt((self.vectors[i, :] ** 2).sum(-1))
                self.vectors_norm = self.vectors
            else:
                self.vectors_norm = (self.vectors / sqrt((self.vectors ** 2).sum(-1))[..., newaxis]).astype(REAL)
```





















