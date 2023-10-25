import random as rand
from WordState import WordState # Use as needed.
    
# Some sentences to test your code on...
test_sentences = ["hello there friend",
                  "hello there good friend",
                  "hello there my good friend",
                  "hello my friend",
                  "hello my good friend",
                  "good day friend",
                  "good morning friend",
                  "good morning to you my friend",
                  "good morning to you my good friend"]
    
# A simple "pre-trained model"/graph derived from above sentences with words and corresponding frequencies.
states = {"#": {"hello": 5, "good": 4},
        "hello": {"there": 3, "my": 2},
        "good": {"friend": 4, "day": 1, "morning": 3},
        "there": {"friend": 1, "good": 1, "my": 1},
        "my": {"good": 2, "friend": 2},
        "friend": {},
        "day": {"friend": 1},
        "morning": {"friend": 1, "to": 2},
        "to": {"you": 2},
        "you": {"my": 1}}
    
# Write your implementation here...


def word_states(sentences: list[str]) -> dict[ dict[str, int] ]:
    """
    Returns a dict of dicts representing the states of each word 
    and their frequencies.
    """

    states = {}

    for sentence in sentences:
        words = sentence.split()
        previous_word = "#"

        for word in words:
                if previous_word not in states:
                    states[previous_word] = {}

                if word not in states[previous_word]:
                    states[previous_word][word] = 1
                else:
                    states[previous_word][word] += 1

                previous_word = word

        if previous_word not in states:
            states[previous_word] = {}

    return states



