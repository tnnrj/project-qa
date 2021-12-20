import re
from numpy import dot
from numpy.linalg import norm



named_entities_by_question = {
    'who': ['PERSON', 'NORP', 'ORG'],
    'where': ['GPE', 'LOC', 'FAC'],
    'when': ['DATE', 'TIME'],
    'how much': ['MONEY', 'QUANTITY', 'PERCENT'],
    'how many': ['CARDINAL'],
    'how long': ['DATE', 'TIME', 'QUANTITY'],
    'how heavy': ['QUANTITY'],
    'how tall': ['QUANTITY'],
    'how old': ['CARDINAL', 'TIME'],
    'percent': ['PERCENT'],
    'cost': ['MONEY'],
    'price': ['MONEY'],
    'title': ['FAC', 'WORK_OF_ART'],
    'name': ['PERSON', 'FAC', 'ORG', 'WORK_OF_ART', 'EVENT', 'PRODUCT'],
    'called': ['PERSON', 'FAC', 'ORG', 'WORK_OF_ART', 'EVENT', 'PRODUCT'],
    'company': ['ORG'],
    'language': ['LANGUAGE'],
    'weigh': ['QUANTITY'],
    'place': ['LOC', 'ORDINAL']
}

def remove_punc(s):
    return re.sub(r'((\!|\.|\?|\,|\"|\'|\:|\;|\)|\]|\}|\-)+\Z)|(^(\"|\(|\[|\{|\-)+)', '', s)

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

    