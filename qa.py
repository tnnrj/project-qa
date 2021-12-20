import sys
import os
import re
import time
import itertools
import gensim
import spacy
from spacy.matcher import Matcher
import helpers

# testset2
# RECALL = .5511
# PRECISION = .3849
# F-MEASURE = .4533

def answer_story_questions(path, story_key):
    story_file = open(path + story_key + ".story")
    story = "".join(story_file.readlines()[6:]).replace('\n', ' ')
    question_file = open(path + story_key + ".questions")
    questions = list((list(g) for k,g in itertools.groupby(question_file.readlines(), key=lambda x: x != '\n') if k))

    doc = nlp(story)
    sents = list(doc.sents)
    N = len(sents)

    out = []
    for q_a in questions:
        question = [helpers.remove_punc(w) for w in q_a[1].strip().lower().split()[1:] if w not in stopwords]
        full_q = " ".join(question)
        q_sent = list(nlp(" ".join(q_a[1].strip().split()[1:])).sents)[0]

        # find desired named entities based on keywords
        desired_ent = set()
        for k in helpers.named_entities_by_question.keys():
            if k in full_q:
                desired_ent.update(helpers.named_entities_by_question[k])

        # assemble question vocab
        q_vocab = set()
        for word in question:
            if word in w2v:
                q_vocab.add(word)
        
        # vector similarity of each word in sentence with question
        scores = []
        contains_ent = [False] * N
        for i, s in enumerate(sents):
            scores.append([0] * len(s))
            for j, word in enumerate(s):
                if word.ent_type_ in desired_ent:
                    contains_ent[i] = True
                elif word.lower_ not in stopwords and word.lower_ in w2v:
                    scores[i][j] = max(map(lambda x: w2v.similarity(word.lower_, x), q_vocab))

        # average score for each sentence
        sent_scores = [0] * N
        for i, s in enumerate(scores):
            sum_ = 0
            len_ = 0
            for num in s:
                if num != 0:
                    sum_ += num
                    len_ += 1
            if len_ != 0:
                sent_scores[i] = sum_ / len_

        # score modifiers
        final_scores = [0] * N
        for i, s in enumerate(sent_scores):
            score = s
            if i > 0 and i < len(sent_scores) - 1:
                score += 0.25*sent_scores[i-1]
                score += 0.25*sent_scores[i+1]
            elif i > 0:
                score += 0.5*sent_scores[i-1]
            elif i < len(sent_scores) - 1:
                score += 0.5*sent_scores[i+1]
            if contains_ent[i]:
                score += .3
            if q_sent.root.lemma_ == sents[i].root.lemma_:
                score += .2
            if "why" in question and "because" in sents[i].lower_:
                score += .3                
            if "when" in question and "when" in sents[i].lower_:
                score += .2
            final_scores[i] = score

        while True:
            # max score
            max_score = max(final_scores)
            max_idx = final_scores.index(max_score)
            answer_sent = sents[max_idx]

            # if answer is an entity
            answer = ''
            for w in answer_sent:
                if w.ent_type_ in desired_ent and not [ent for ent in doc.ents if ent.start_char <= w.idx <= ent.end_char][0].lower_ in full_q:
                    if w.ent_type_ == "MONEY" and w.left_edge.is_currency and not w.left_edge.ent_type_ in desired_ent:
                        answer += w.left_edge.text
                    answer += w.text + ' '

            # otherwise return the whole sentence
            if answer == '':
                answer = answer_sent.text
                if not answer.endswith('?'):
                    break
                final_scores[max_idx] = 0
            else:
                answer_story_questions.ner_count += 1
                break

        # hand-written rules
        if "why" in question and "because" in answer_sent.lower_:
            answer = "because" + answer_sent.lower_.split("because")[1]
        if "where" in question and " at " in answer:
            answer = "at " + answer.split(" at ")[1]
        if "where" in question and " in " in answer:
            answer = "in " + answer.split(" in ")[1]

        for word in answer.split():
            try:
                if w2v.vocab[word.lower()].count < 299900 and re.search(r'(\s+|^)' + word.lower() + r'(\s+|$)', q_sent.lower_) is not None:
                    answer = re.sub(r'(\s+|^)' + word + r'(\s+|$)', ' ', answer)
            except:
                pass        

        if os.name == 'nt':
            out.append(q_a[0])
            out.append(f"Answer: {answer}\n\n")
        else:
            print(q_a[0])
            print(f"Answer: {answer}\n\n")

    if os.name == 'nt':
        out_file.writelines(out)


# entry point

if os.name == 'nt':
    start_time = time.perf_counter()

nlp = spacy.load('en_core_web_lg')

if os.name == 'nt':
    stopwords = {w.strip() for w in open("./data/training/stopwords.txt").readlines()}
    w2v = gensim.models.KeyedVectors.load("./data/training/word2vec.vectors", mmap='r')
    out_file = open("all.answers", "w")
else:
    stopwords = {w.strip() for w in open("./data/training/stopwords.txt").readlines()}
    w2v = gensim.models.KeyedVectors.load("/home/u0907330/QA_data/word2vec.vectors", mmap='r')

infile_name = sys.argv[1]
infile = open(infile_name)

lines = infile.readlines()
path = lines[0].strip()
if not path.endswith('/'):
    path += '/'

if os.name == 'nt':
    print(f"Startup time: {time.perf_counter() - start_time}")
    start_time = time.perf_counter()

answer_story_questions.ner_count = 0

for story_key in lines[1:]:
    if story_key.startswith('#'):
        break
    answer_story_questions(path, story_key.strip())

if os.name == 'nt':
    total = time.perf_counter() - start_time
    print(f"Total time: {total}")
    print(f"Time per story: {total / (len(lines)-1)}")
    print(f"NER: {answer_story_questions.ner_count}")
    out_file.close()

