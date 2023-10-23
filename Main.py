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

def generate_sentence(states):
        """Generates a random sentence based on the given states."""
        sentence = []
        current_word = "#"
        while True:
                if not states[current_word]:
                        break
                next_word = rand.choices(list(states[current_word].keys()), list(states[current_word].values()))[0]
                if next_word == "":
                        break
                sentence.append(next_word)
                current_word = next_word
        return " ".join(sentence) 


# Test your code here...
print(generate_sentence(states))