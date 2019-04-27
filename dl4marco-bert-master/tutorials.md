code:
https://github.com/nyu-dl/dl4marco-bert

### Annotated code
convert_msmarco_to_tfrecord.py
run_msmarco.py
run_treccar.py

### notes
The model is trained using a classification loss (cross entropy loss) to classify if a document is relevant or not for a given query. It outputs a probability of a document being relevant for a given query. The documents are re-ranked based on the probability assigned to each one of them.
Since the documents are classified independently, the model and the loss do not consider multiple documents.[site](https://github.com/nyu-dl/dl4marco-bert/issues/8) 

