import os
import random
import re
import sys
from collections import Counter

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = dict()
    page = ''.join(list(page))
    if len(corpus[page])==0:
        for i in corpus:
            model[i] = 1/len(corpus)
        return model
    for i in corpus:
        if i not in corpus[i]:
            model[i] = 0
    p = 1/len(corpus[page])
    for j in corpus[page]:
        model[j] = damping_factor*p
    l = len(model.keys())
    for key in model:
        prob = (1-damping_factor)
        model[key] += prob/l
    return model

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    dist = [1/len(corpus) for i in range(len(corpus))]
    pop = list(corpus.keys())
    sample = [''.join(list(random.choices(pop,weights=dist)))]
    for j in range(n-1):
        model = transition_model(corpus,sample[-1],damping_factor)
        dist = list(model.values())
        pop = list(model.keys())
        next_page = ''.join(list(random.choices(pop,weights=dist)))
        sample.append(next_page)
    sample = Counter(sample)
    for i in sample:
        sample[i] /= n
    return sample


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(list(corpus.keys()))
    model = {key:1/N for key in corpus}
    for key in corpus:
        if len(corpus[key]) == 0:
            corpus[key] = set(corpus.keys())
    flag = 1
    while flag:
        flag = 0
        for key in corpus:
            summation = 0
            for i in corpus:
                if key in corpus[i]:
                    PR = model[i]
                    NL = len(corpus[i])
                    summation+=(PR/NL)
            new = (1-damping_factor)/N + damping_factor*summation
            if abs(model[key]-new) > 0.001:
                flag = 1
            model[key] = new
    return model



if __name__ == "__main__":
    main()
