import random as rand
import pandas as pd
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


def word_states(sentences: list) -> dict:
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


#consider breaking up into smaller functions
def generate_sentences(sentences, num_sentences):
    states = {"#": WordState()}

    for sentence in sentences:
        words = sentence.split()
        previous_word = "#"

        for word in words:
            if previous_word not in states:
                states[previous_word] = WordState()

            states[previous_word].add_next_word(word)

            previous_word = word

        #last word is not followed by anything
        if previous_word not in states:
            states[previous_word] = WordState()

    sentences = []

    for i in range(num_sentences):
        sentence = []
        current_word = "#"
        while True:
            sentence.append(current_word)
            if not states[current_word].has_next():
                break
            current_word = states[current_word].get_next()
        sentences.append(" ".join(sentence[1:]))

    return sentences


def preprocess_text(
        data: pd.DataFrame, text_column: str,
        lower_casing               : bool = False, 
        remove_punctuations        : bool = False,
        remove_stopwords           : bool = False, 
        frequent_words_to_remove   : int  = 0,
        rare_words_to_remove       : int  = 0,
        remove_emojis              : bool = True,
        remove_emoticons           : bool = True,
        convert_emoticons_to_words : bool = False,
        remove_urls                : bool = True
):
    text = data[text_column].copy()

    if lower_casing:
        text = text.str.lower()
    
    if remove_punctuations:
        import string
        text = text.apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation))
        )

    if remove_stopwords:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        #Test later if this is necessary
        ", ".join(stopwords.words('english'))

        STOPWORDS = set(stopwords.words('english'))
        text = text.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in STOPWORDS]
            )
        )

    from collections import Counter

    if frequent_words_to_remove > 0:
        counter = Counter()
        for value in text.values:
            for word in value.split():
                counter[word] += 1
        
        FREQUENT_WORDS = set(
            [
                word for (word, count) 
                in counter.most_common(frequent_words_to_remove)
            ]
        )

        text = text.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in FREQUENT_WORDS]
            )
        )

    if rare_words_to_remove > 0:
        counter = Counter()
        for value in text.values:
            for word in value.split():
                counter[word] += 1
        
        RARE_WORDS = set(
            [
                word for (word, count) 
                in counter.most_common()[:-rare_words_to_remove-1:-1]
            ]
        )

        text = text.apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in RARE_WORDS]
            )
        )

    import re
    
    if remove_emojis:
        emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
        
        text = text.apply(lambda x: emoji_pattern.sub(r'', x))

    EMOTICONS = {
        u":‑\)":"Happy face or smiley",
        u":\)":"Happy face or smiley",
        u":-\]":"Happy face or smiley",
        u":\]":"Happy face or smiley",
        u":-3":"Happy face smiley",
        u":3":"Happy face smiley",
        u":->":"Happy face smiley",
        u":>":"Happy face smiley",
        u"8-\)":"Happy face smiley",
        u":o\)":"Happy face smiley",
        u":-\}":"Happy face smiley",
        u":\}":"Happy face smiley",
        u":-\)":"Happy face smiley",
        u":c\)":"Happy face smiley",
        u":\^\)":"Happy face smiley",
        u"=\]":"Happy face smiley",
        u"=\)":"Happy face smiley"
    }
    
    if remove_emoticons:
        emoticon_pattern = re.compile(
            u'(' + u'|'.join(k for k in EMOTICONS) + u')'
        )
        text = text.apply(
            lambda x: emoticon_pattern.sub(r'', x)
        )

    if convert_emoticons_to_words:
        for emoticon in EMOTICONS:
            text = re.sub(
                u'('+emoticon+')',
                "_".join(EMOTICONS[emoticon].replace(",","").split()),
                text
            )
    
    if remove_urls:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text = text.apply(
            lambda x: url_pattern.sub(r'', x)
        )
    
    return text


def sentences_from_poems(poems : pd.DataFrame) -> list:
    """
    Returns a list of sentences from the poems.
    """
    sentences = []

    for poem in poems:
        sentences.append(poem.split("\n"))
    
    #list of lists to list of strings
    sentences = [sentence for poem in sentences for sentence in poem]

    return sentences


def process_output_poems(poems : list) -> list:
    new_poems = []
    for poem in poems:
        poem = poem.replace(".", ".\n")
        poem = poem.replace(",", ",\n")
        poem = poem.replace("?", "?\n")
        poem = poem.replace("!", "!\n")
        #new line for words starting with capital letter
        poem = poem.replace(" ", "\n", 1)
        new_poems.append(poem)
    
    return new_poems


def generate_poems(csv, column, amount_of_poems):
    data = pd.read_csv(csv)
    data = preprocess_text(data, column)
    data = sentences_from_poems(data)
    poems = generate_sentences(data, amount_of_poems)
    poems = process_output_poems(poems)

    for poem in poems:
        print("##############################################################")
        print(poem)
        print("##############################################################")
    
    return poems


def save_generated_text(text, filename):
    print("Saving generated text to file: " + filename)
    with open(filename, "w", encoding="utf8") as file:
        for line in text:
            file.write(line + "\n")
    print("Text saved to file: " + filename + "\n")
    

def main():
    poem_data = "PoetryFoundationData.csv"
    poem_column = "Poem"

    keep_running = True
    print("--- Poem Generator ---")

    while keep_running:
        print("Press 1 to generate poems. 0 to exit.")
        print()
        user_input = input("Enter your choice: ")

        if user_input == "1":
            number_of_poems = int( input("Number of poems to generate: ") )
            poems = generate_poems(poem_data, poem_column, number_of_poems)
            save_poems = input("Save poems to file? (y/n): ")
            if save_poems == "y":
                try:
                    file_name = input("Enter file name: ")
                    save_generated_text(poems, file_name)
                except:
                    print("Error saving poems to file. (check file name)")
            elif save_poems == "n":
                pass
        

if __name__ == "__main__":
    main()