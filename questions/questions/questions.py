import string

import nltk
import sys
import os
import math

nltk.download('stopwords')
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    os.chdir(directory)
    d = {}
    for file in files:
        with open(file, encoding = "utf-8") as f:
            d[file] = f.read()
    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punc = string.punctuation
    stop = nltk.corpus.stopwords.words("english")
    words = nltk.word_tokenize(document.lower())
    pro = [word for word in words if word not in punc and word not in stop]
    return pro


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set(sum(documents.values(),[]))
    totdoc = len(documents)
    idf = {}
    for word in words:
        count = 0
        for d in documents.values():
            if word in d:
                count+=1
        idf[word] = math.log(totdoc/count)
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = {}
    for file,content in files.items():
        filetfidf = 0
        for word in query:
            if word in content:
                filetfidf += content.count(word)*idfs[word]
        if filetfidf:
            tfidf[file] = filetfidf
    return [key for key,value in sorted(tfidf.items(), key=lambda tup:tup[-1], reverse=True)][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = {}
    for sent, words in sentences.items():
        sentidf = 0
        for word in query:
            if word in words:
                sentidf += idfs[word]
        if sentidf:
            idf[sent] = (sentidf, len(set(words).union(query))/len(query))
    return [key for key, value in sorted(idf.items(), key=lambda tup: (tup[-1][0],tup[-1][-1]), reverse=True)][:n]


if __name__ == "__main__":
    main()
