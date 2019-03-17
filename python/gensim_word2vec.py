'''
online word2vec
https://github.com/RaRe-Technologies/gensim/pull/365

Speed up load word2vec
https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time
1. load_word2vec_format() with limit parameter
2. following steps:
First, load the word2vec.c-format vectors, with load_word2vec_format().
Then, use model.init_sims(replace=True) to force the unit-normalization, destructively in-place (clobbering the non-normalized vectors).

Then, save the model to a new filename-prefix: model.save('GoogleNews-vectors-gensim-normed.bin'`.
(Note that this actually creates multiple files on disk that need to be kept together for the model to be re-loaded.)

Then, We also want this program to hang until externally terminated (keeping the mapping alive),
and be careful not to re-calculate the already-normed vectors.
from gensim.models import KeyedVectors
from threading import Semaphore
model = KeyedVectors.load('GoogleNews-vectors-gensim-normed.bin', mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
# model.most_similar('stuff')  # any word will do: just to page all in
Semaphore(0).acquire()  # just hang until process killed

If the system is facing other memory pressure, ranges of the array may fall out of memory until
the next read pages them back in. And if the machine lacks the RAM to ever fully load the vectors,
then every scan will require a mixing of paging-in-and-out, and performance will be frustratingly
bad not matter what. (In such a case: get more RAM or work with a smaller vector set.)

3. https://github.com/3Top/word2vec-api
'''

# How to pre-allocate the memory for the word2vec?

