# project-qa
A rule-based information extraction Python script to answer questions about sample stories

## inspiration
Builds upon work done by Dr. Ellen Riloff in [A Rule-based Question Answering System for Reading Comprehension Tests](https://www.cs.utah.edu/~riloff/pdfs/quarc.pdf), with marginal improvements in precision and recall.

## setup
Commands in included QA-script.txt should be run to install relevant packages and download spaCy corpus.
Gensim relies on preconstructed [word2vec](https://code.google.com/archive/p/word2vec) vectors. These can be built from any sufficiently large corpus, but in this case I recommend the [Google News dataset](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM). They should be parsed into Gensim's `KeyedVectors` model and saved in `./data/training/word2vec.vectors`.

## approach
This script assigns a score to each sentence based on:
 - Distributional similarity of word embeddings
 - Named entity recognition
 - Sentence root identification

Then, attempts to filter the sentence down to relevant noun phrases or named entities.
Detailed explanation contained in [these slides](https://docs.google.com/presentation/d/17C0Ydxh-qrx1wxoOcOgdqoaY-lh3Fm16G3g9pSN_UXQ)


## testing
This script is built to answer questions of any type given a relevant document. Sample input formats are shown in `data/testset1`. A scoring program has been included to output precision, recall, and F-score of produced answers.

## performance
Running QA requires significant memory overhead as it loads w2v and spaCy corpus into memory before beginning. Afterwards reading and answering each story takes runtime on the order of one second, the majority of which is parsing the story into a spaCy nlp model. On a final test set of documents and questions, this system produced the following results:
 - RECALL = 0.5511
 - PRECISION = 0.3849
 - F-MEASURE = 0.4533

## tools
 - spaCy : https://spacy.io/
    for named entity recognition and token parsing
 - gensim : https://radimrehurek.com/gensim/
    for word2vec vector comparison
    
Special thanks to [Dr. Ellen Riloff](http://www.cs.utah.edu/~riloff/) at the University of Utah.
