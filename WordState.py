import random as rand

class WordState:
    """
    Helper class representing a word and all its next words and frequencies.
    """

    def __init__(self):
        self._next_words = {} # A dict with all the next words and their frequencies.
        

    def add_next_word(self, next_word):
        """
        Introduces a new next word to the current word.
        If the word already exists in the dict, its count is incremented.
        """

        if next_word in self._next_words:
            self._next_words[next_word] += 1
        else:
            self._next_words[next_word] = 1


    def has_next(self):
        """True if there are any more words following this one."""

        return bool(self._next_words)
        

    def get_next(self):
        """Returns a random next word based on probability."""

        return rand.choices(list(self._next_words.keys()), weights=list(self._next_words.values()))[0]

            