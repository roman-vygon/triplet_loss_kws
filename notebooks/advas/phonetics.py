# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# AdvaS Advanced Search 0.2.5
# advanced search algorithms implemented as a python module
# phonetics module
#
# (C) 2002 - 2013 Frank Hofmann, Berlin, Germany
# Released under GNU Public License (GPL)
# email fh@efho.de
# -----------------------------------------------------------

import re

from .ngram import Ngram


class Phonetics:
    def __init__(self, term):
        self.term = term
        return

    def setText(self, term):
        self.term = term
        return

    def getText(self):
        return self.term

    # covering algorithms

    def phoneticCode(self):
        "returns the term's phonetic code using different methods"

        # build an array to hold the phonetic code for each method
        phoneticCodeList = {
            "soundex": self.soundex(),
            "metaphone": self.metaphone(),
            "nysiis": self.nysiis(),
            "caverphone": self.caverphone()
        }

        return phoneticCodeList

    # phonetic algorithms

    def soundex(self):
        "Return the soundex value to a given string."

        # Create and compare soundex codes of English words.
        #
        # Soundex is an algorithm that hashes English strings into
        # alpha-numerical value that represents what the word sounds
        # like. For more information on soundex and some notes on the
        # differences in implemenations visit:
        # http://www.bluepoof.com/Soundex/info.html
        #
        # This version modified by Nathan Heagy at Front Logic Inc., to be
        # compatible with php's soundexing and much faster.
        #
        # eAndroid / Nathan Heagy / Jul 29 2000
        # changes by Frank Hofmann / Jan 02 2005, Sep 9 2012

        # generate translation table only once. used to translate into soundex numbers
        # table = string.maketrans('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123012002245501262301020201230120022455012623010202')
        table = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', '01230120022455012623010202')

        # check parameter
        if not self.term:
            return "0000"  # could be Z000 for compatibility with other implementations

        # convert into uppercase letters
        term = str.upper(self.term)
        firstChar = term[0]

        # translate the string into soundex code according to the table above
        term = str.translate(term[1:], table)

        # remove all 0s
        term = str.replace(term, "0", "")

        # remove duplicate numbers in-a-row
        str2 = firstChar
        for x in term:
            if x != str2[-1]:
                str2 = str2 + x

        # pad with zeros
        str2 = str2 + "0" * len(str2)

        # return the first four letters
        return str2[:4]

    def metaphone(self):
        "returns metaphone code for a given string"

        # implementation of the original algorithm from Lawrence Philips
        # extended/rewritten by M. Kuhn
        # improvements with thanks to John Machin <sjmachin@lexicon.net>

        # define return value
        code = ""

        i = 0
        termLength = len(self.term)

        if (termLength == 0):
            # empty string ?
            return code

        # extension #1 (added 2005-01-28)
        # convert to lowercase
        term = str.lower(self.term)

        # extension #2 (added 2005-01-28)
        # remove all non-english characters, first
        term = re.sub(r'[^a-z]', '', term)
        if len(term) == 0:
            # nothing left
            return code

        # extension #3 (added 2005-01-24)
        # conflate repeated letters
        firstChar = term[0]
        str2 = firstChar
        for x in term:
            if x != str2[-1]:
                str2 = str2 + x

        # extension #4 (added 2005-01-24)
        # remove any vowels unless a vowel is the first letter
        firstChar = str2[0]
        str3 = firstChar
        for x in str2[1:]:
            if (re.search(r'[^aeiou]', x)):
                str3 = str3 + x

        term = str3
        termLength = len(term)
        if termLength == 0:
            # nothing left
            return code

        # check for exceptions
        if (termLength > 1):
            # get first two characters
            firstChars = term[0:2]

            # build translation table
            table = {
                "ae": "e",
                "gn": "n",
                "kn": "n",
                "pn": "n",
                "wr": "n",
                "wh": "w"
            }

            if firstChars in table.keys():
                term = term[2:]
                code = table[firstChars]
                termLength = len(term)

        elif (term[0] == "x"):
            term = ""
            code = "s"
            termLength = 0

        # define standard translation table
        stTrans = {
            "b": "b",
            "c": "k",
            "d": "t",
            "g": "k",
            "h": "h",
            "k": "k",
            "p": "p",
            "q": "k",
            "s": "s",
            "t": "t",
            "v": "f",
            "w": "w",
            "x": "ks",
            "y": "y",
            "z": "s"
        }

        i = 0
        while (i < termLength):
            # init character to add, init basic patterns
            add_char = ""
            part_n_2 = ""
            part_n_3 = ""
            part_n_4 = ""
            part_c_2 = ""
            part_c_3 = ""

            # extract a number of patterns, if possible
            if (i < (termLength - 1)):
                part_n_2 = term[i:i + 2]

                if (i > 0):
                    part_c_2 = term[i - 1:i + 1]
                    part_c_3 = term[i - 1:i + 2]

            if (i < (termLength - 2)):
                part_n_3 = term[i:i + 3]

            if (i < (termLength - 3)):
                part_n_4 = term[i:i + 4]

            # use table with conditions for translations
            if (term[i] == "b"):
                addChar = stTrans["b"]
                if (i == (termLength - 1)):
                    if (i > 0):
                        if (term[i - 1] == "m"):
                            addChar = ""
            elif (term[i] == "c"):
                addChar = stTrans["c"]
                if (part_n_2 == "ch"):
                    addChar = "x"
                elif (re.search(r'c[iey]', part_n_2)):
                    addChar = "s"

                if (part_n_3 == "cia"):
                    addChar = "x"

                if (re.search(r'sc[iey]', part_c_3)):
                    addChar = ""

            elif (term[i] == "d"):
                addChar = stTrans["d"]
                if (re.search(r'dg[eyi]', part_n_3)):
                    addChar = "j"

            elif (term[i] == "g"):
                addChar = stTrans["g"]

                if (part_n_2 == "gh"):
                    if (i == (termLength - 2)):
                        addChar = ""
                elif (re.search(r'gh[aeiouy]', part_n_3)):
                    addChar = ""
                elif (part_n_2 == "gn"):
                    addChar = ""
                elif (part_n_4 == "gned"):
                    addChar = ""
                elif (re.search(r'dg[eyi]', part_c_3)):
                    addChar = ""
                elif (part_n_2 == "gi"):
                    if (part_c_3 != "ggi"):
                        addChar = "j"
                elif (part_n_2 == "ge"):
                    if (part_c_3 != "gge"):
                        addChar = "j"
                elif (part_n_2 == "gy"):
                    if (part_c_3 != "ggy"):
                        addChar = "j"
                elif (part_n_2 == "gg"):
                    addChar = ""
            elif (term[i] == "h"):
                addChar = stTrans["h"]
                if (re.search(r'[aeiouy]h[^aeiouy]', part_c_3)):
                    addChar = ""
                elif (re.search(r'[csptg]h', part_c_2)):
                    addChar = ""
            elif (term[i] == "k"):
                addChar = stTrans["k"]
                if (part_c_2 == "ck"):
                    addChar = ""
            elif (term[i] == "p"):
                addChar = stTrans["p"]
                if (part_n_2 == "ph"):
                    addChar = "f"
            elif (term[i] == "q"):
                addChar = stTrans["q"]
            elif (term[i] == "s"):
                addChar = stTrans["s"]
                if (part_n_2 == "sh"):
                    addChar = "x"
                if (re.search(r'si[ao]', part_n_3)):
                    addChar = "x"
            elif (term[i] == "t"):
                addChar = stTrans["t"]
                if (part_n_2 == "th"):
                    addChar = "0"
                if (re.search(r'ti[ao]', part_n_3)):
                    addChar = "x"
            elif (term[i] == "v"):
                addChar = stTrans["v"]
            elif (term[i] == "w"):
                addChar = stTrans["w"]
                if (re.search(r'w[^aeiouy]', part_n_2)):
                    addChar = ""
            elif (term[i] == "x"):
                addChar = stTrans["x"]
            elif (term[i] == "y"):
                addChar = stTrans["y"]
            elif (term[i] == "z"):
                addChar = stTrans["z"]
            else:
                # alternative
                addChar = term[i]
            code = code + addChar
            i += 1
        # end while

        return code

    def nysiis(self):
        "returns New York State Identification and Intelligence Algorithm (NYSIIS) code for the given term"

        code = ""

        i = 0
        term = self.term
        termLength = len(term)

        if (termLength == 0):
            # empty string ?
            return code

        # build translation table for the first characters
        table = {
            "mac": "mcc",
            "ph": "ff",
            "kn": "nn",
            "pf": "ff",
            "k": "c",
            "sch": "sss"
        }

        for tableEntry in table.keys():
            tableValue = table[tableEntry]  # get table value
            tableValueLen = len(tableValue)  # calculate its length
            firstChars = term[0:tableValueLen]
            if (firstChars == tableEntry):
                term = tableValue + term[tableValueLen:]
                break

        # build translation table for the last characters
        table = {
            "ee": "y",
            "ie": "y",
            "dt": "d",
            "rt": "d",
            "rd": "d",
            "nt": "d",
            "nd": "d",
        }

        for tableEntry in table.keys():
            tableValue = table[tableEntry]  # get table value
            tableEntryLen = len(tableEntry)  # calculate its length
            lastChars = term[(0 - tableEntryLen):]
            # print lastChars, ", ", tableEntry, ", ", tableValue
            if (lastChars == tableEntry):
                term = term[:(0 - tableValueLen + 1)] + tableValue
                break

        # initialize code
        code = term

        # transform ev->af
        code = re.sub(r'ev', r'af', code)

        # transform a,e,i,o,u->a
        code = re.sub(r'[aeiouy]', r'a', code)

        # transform q->g
        code = re.sub(r'q', r'g', code)

        # transform z->s
        code = re.sub(r'z', r's', code)

        # transform m->n
        code = re.sub(r'm', r'n', code)

        # transform kn->n
        code = re.sub(r'kn', r'n', code)

        # transform k->c
        code = re.sub(r'k', r'c', code)

        # transform sch->sss
        code = re.sub(r'sch', r'sss', code)

        # transform ph->ff
        code = re.sub(r'ph', r'ff', code)

        # transform h-> if previous or next is nonvowel -> previous
        occur = re.findall(r'([a-z]{0,1}?)h([a-z]{0,1}?)', code)
        # print occur
        for occurGroup in occur:
            occurItemPrevious = occurGroup[0]
            occurItemNext = occurGroup[1]

            if ((re.match(r'[^aeiouy]', occurItemPrevious)) or (re.match(r'[^aeiouy]', occurItemNext))):
                if (occurItemPrevious != ""):
                    # make substitution
                    code = re.sub(occurItemPrevious + "h", occurItemPrevious * 2, code, 1)

        # transform w-> if previous is vowel -> previous
        occur = re.findall(r'([aeiouy]{1}?)w', code)
        # print occur
        for occurGroup in occur:
            occurItemPrevious = occurGroup[0]
            # make substitution
            code = re.sub(occurItemPrevious + "w", occurItemPrevious * 2, code, 1)

        # check last character
        # -s, remove
        code = re.sub(r's$', r'', code)
        # -ay, replace by -y
        code = re.sub(r'ay$', r'y', code)
        # -a, remove
        code = re.sub(r'a$', r'', code)

        return code

    def caverphone(self):
        "returns the language key using the caverphone algorithm 2.0"

        # Developed at the University of Otago, New Zealand.
        # Project: Caversham Project (http://caversham.otago.ac.nz)
        # Developer: David Hood, University of Otago, New Zealand
        # Contact: caversham@otago.ac.nz
        # Project Technical Paper: http://caversham.otago.ac.nz/files/working/ctp150804.pdf
        # Version 2.0 (2004-08-15)

        code = ""

        i = 0
        term = self.term
        termLength = len(term)

        if (termLength == 0):
            # empty string ?
            return code

        # convert to lowercase
        code = str.lower(term)

        # remove anything not in the standard alphabet (a-z)
        code = re.sub(r'[^a-z]', '', code)

        # remove final e
        if code.endswith("e"):
            code = code[:-1]

        # if the name starts with cough, rough, tough, enough or trough -> cou2f (rou2f, tou2f, enou2f, trough)
        code = re.sub(r'^([crt]|(en)|(tr))ough', r'\1ou2f', code)

        # if the name starts with gn -> 2n
        code = re.sub(r'^gn', r'2n', code)

        # if the name ends with mb -> m2
        code = re.sub(r'mb$', r'm2', code)

        # replace cq -> 2q
        code = re.sub(r'cq', r'2q', code)

        # replace c[i,e,y] -> s[i,e,y]
        code = re.sub(r'c([iey])', r's\1', code)

        # replace tch -> 2ch
        code = re.sub(r'tch', r'2ch', code)

        # replace c,q,x -> k
        code = re.sub(r'[cqx]', r'k', code)

        # replace v -> f
        code = re.sub(r'v', r'f', code)

        # replace dg -> 2g
        code = re.sub(r'dg', r'2g', code)

        # replace ti[o,a] -> si[o,a]
        code = re.sub(r'ti([oa])', r'si\1', code)

        # replace d -> t
        code = re.sub(r'd', r't', code)

        # replace ph -> fh
        code = re.sub(r'ph', r'fh', code)

        # replace b -> p
        code = re.sub(r'b', r'p', code)

        # replace sh -> s2
        code = re.sub(r'sh', r's2', code)

        # replace z -> s
        code = re.sub(r'z', r's', code)

        # replace initial vowel [aeiou] -> A
        code = re.sub(r'^[aeiou]', r'A', code)

        # replace all other vowels [aeiou] -> 3
        code = re.sub(r'[aeiou]', r'3', code)

        # replace j -> y
        code = re.sub(r'j', r'y', code)

        # replace an initial y3 -> Y3
        code = re.sub(r'^y3', r'Y3', code)

        # replace an initial y -> A
        code = re.sub(r'^y', r'A', code)

        # replace y -> 3
        code = re.sub(r'y', r'3', code)

        # replace 3gh3 -> 3kh3
        code = re.sub(r'3gh3', r'3kh3', code)

        # replace gh -> 22
        code = re.sub(r'gh', r'22', code)

        # replace g -> k
        code = re.sub(r'g', r'k', code)

        # replace groups of s,t,p,k,f,m,n by its single, upper-case equivalent
        for singleLetter in ["s", "t", "p", "k", "f", "m", "n"]:
            otherParts = re.split(singleLetter + "+", code)
            code = str.upper(singleLetter).join(otherParts)

        # replace w[3,h3] by W[3,h3]
        code = re.sub(r'w(h?3)', r'W\1', code)

        # replace final w with 3
        code = re.sub(r'w$', r'3', code)

        # replace w -> 2
        code = re.sub(r'w', r'2', code)

        # replace h at the beginning with an A
        code = re.sub(r'^h', r'A', code)

        # replace all other occurrences of h with a 2
        code = re.sub(r'h', r'2', code)

        # replace r3 with R3
        code = re.sub(r'r3', r'R3', code)

        # replace final r -> 3
        code = re.sub(r'r$', r'3', code)

        # replace r with 2
        code = re.sub(r'r', r'2', code)

        # replace l3 with L3
        code = re.sub(r'l3', r'L3', code)

        # replace final l -> 3
        code = re.sub(r'l$', r'3', code)

        # replace l with 2
        code = re.sub(r'l', r'2', code)

        # remove all 2's
        code = re.sub(r'2', r'', code)

        # replace the final 3 -> A
        code = re.sub(r'3$', r'A', code)

        # remove all 3's
        code = re.sub(r'3', r'', code)

        # extend the code by 10 '1' (one)
        code += '1' * 10

        # return the first 10 characters
        return code[:10]

    def calcSuccVariety(self):

        # derive two-letter combinations
        ngramObject = Ngram(self.term, 2)
        ngramObject.deriveNgrams()
        ngramSet = set(ngramObject.getNgrams())

        # count appearances of the second letter
        varietyList = {}
        for entry in ngramSet:
            letter1 = entry[0]
            letter2 = entry[1]
            if varietyList.has_key(letter1):
                items = varietyList[letter1]
                if not letter2 in items:
                    # extend the existing one
                    items.append(letter2)
                    varietyList[letter1] = items
            else:
                # create a new one
                varietyList[letter1] = [letter2]

        return varietyList

    def calcSuccVarietyCount(self, varietyList):
        # save the number of matches, only
        for entry in varietyList:
            items = len(varietyList[entry])
            varietyList[entry] = items
        return varietyList

    def calcSuccVarietyList(self, wordList):
        result = {}
        for item in wordList:
            self.setText(item)
            varietyList = self.calcSuccVariety()
            result[item] = varietyList

        return result

    def calcSuccVarietyMerge(self, varietyList):
        result = {}
        for item in varietyList.values():
            for letter in item.keys():
                if not letter in result.keys():
                    result[letter] = item[letter]
                else:
                    result[letter] = list(set(result[letter]) | set(item[letter]))
        return result
