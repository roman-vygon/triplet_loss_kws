# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# AdvaS Advanced Search 0.2.5
# advanced search algorithms implemented as a python module
# ngram module
#
# (C) 2002 - 2012 Frank Hofmann, Berlin, Germany
# Released under GNU Public License (GPL)
# email fh@efho.de
# -----------------------------------------------------------

class Ngram:

    def __init__(self, term, size):
        self.term = term
        self.setNgramSize(size)
        self.ngrams = []
        return

    def getNgramSize(self):
        return self.ngramSize

    def setNgramSize(self, size):
        self.ngramSize = size
        return

    def getNgrams(self):
        return self.ngrams

    def deriveNgrams(self):
        "derive n-grams of size n"

        termLength = len(self.term)

        if (self.ngramSize > termLength):
            # we cannot form any n-grams - term too small for given size
            self.ngrams = []
            return False

        # define left and right boundaries
        left = 0
        right = left + self.ngramSize

        while (right <= termLength):
            # extract slice and append to the list
            ngram = self.term[left:right]
            if not ngram in self.ngrams:
                self.ngrams.append(ngram)

            # move slice to the right
            left = left + 1
            right = right + 1

        return True
