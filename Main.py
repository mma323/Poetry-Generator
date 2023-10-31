import random as rand
import pandas as pd
from WordState import WordState
import os

DEFAULT_DATA = "PoetryFoundationData.csv"
DEFAULT_COLUMN = "Poem"

def main():
    data = DEFAULT_DATA
    column = DEFAULT_COLUMN
    debug=True
    if debug:
        pass

    keep_running = True
    while keep_running:
        print("\n--- Poem Generator ---")
        print("Current training data source: " + data + "\n")

        try:
            if data == DEFAULT_DATA:
                user_input = display_menu_for_default_data()
                if user_input == "1":
                    generate_poems_for_default_data(data, column)
                elif user_input == "2":
                    data = change_training_data_source()
                elif user_input == "0":
                    keep_running = False
            else:
                user_input = display_menu_for_non_default_data()
                if user_input == "1":
                    generate_sentences_for_non_default_data(data)
                elif user_input == "2":
                    data = change_training_data_source()
                elif user_input == "0":
                    keep_running = False
        except FileNotFoundError:
            print("File not found. Try again.")

    
def read_and_parse_text(file: str, column=None, category=None) -> pd.DataFrame:
    """
    Reads and parses text from a CSV file.

    Args:
    file (str): The path to the CSV file.
    column (str): The name of the column to read from. If None, reads from the first column.
    category (str): The category to filter by. If None, returns all rows.

    Returns:
    pd.DataFrame: The parsed text.
    """

    if column is None:
        text = pd.read_csv(file, header=None)
        text = text[0]
        return text
    else:
        text = pd.read_csv(file)
        if category is not None:
            text = text[text['Tags'].str.contains(category, na=False)][column]
        else:
            text = text[column]
        return text
   

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
def generate_sentences(sentences : list , num_sentences : int):
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
        data: pd.DataFrame, 
        lower_casing               : bool = False, 
        remove_punctuations        : bool = False,
        remove_stopwords           : bool = False, 
        frequent_words_to_remove   : int  = 0,
        rare_words_to_remove       : int  = 0,
        remove_emojis              : bool = False,
        remove_emoticons           : bool = False,
        convert_emoticons_to_words : bool = False,
        remove_urls                : bool = False
):
    text = data.copy()

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


def process_output_poems(poems : list,) -> list:
    new_poems = []
    characters_ending_line = [".", ",", "?", "!", ";", ":"]
    characters_to_be_removed = ["-", "_", "—", "–", "(", ")", '"', "“", "”"]

    for poem in poems:
        for character in characters_ending_line:
            poem = poem.replace(character, character + " \n")

        for character in characters_to_be_removed:
            poem = poem.replace(character, "")
        
        #remove whitespace in the beginning of each line
        poem = "\n".join(
            [line.lstrip() for line in poem.split("\n")]
        )

        #remove lines containing more than 10 words
        poem = "\n".join(
            [line for line in poem.split("\n") if len(line.split()) < 10]
        )
        
        new_poems.append(poem)
        
    return new_poems


def generate_poems(csv, column, amount_of_poems):
    data = read_and_parse_text(csv, column)
    data = preprocess_text(data)
    data = sentences_from_poems(data)
    poems = generate_sentences(data, amount_of_poems)
    poems = process_output_poems(poems)

    for poem in poems:
        print("##############################################################")
        print(poem)
        print("##############################################################")
    
    return poems


def save_generated_text(text):
    try:
        file = input("Enter file name: ")

        while os.path.exists(file):
            print("File already exists. Are you sure you want to overwrite it?")
            overwrite = input("Enter choice (y/n): ")
            if overwrite == "y":
                break
            elif overwrite == "n":
                file = input("Enter new file name: ")
            else:
                print("Invalid input. Try again.")
                overwrite = input("Enter choice (y/n): ")
            
        print("Saving generated text to file: " + file)
        with open(file, "w", encoding="utf8") as file:
            for line in text:
                file.write(line + "\n")
        print("Text saved to file: " + file + "\n")
    except:
        print("Error saving text to file. (check file name/format)")
        

def display_menu_for_default_data():
    print(
        f"Press 1 to generate poem(s) \n"
        "Press 2 to change training data source.\n"
        "Press 0 to exit. \n"
    )
    return input("Enter choice: ")


def display_menu_for_non_default_data():
    print(
        f"Press 1 to generate sentences \n"
        "Press 2 to change training data source.\n"
        "Press 0 to exit. \n"
    )
    return input("Enter choice: ")


def generate_poems_for_default_data(data, column):
    number_of_poems = int( input("Number of poems to generate: ") )
    poems = generate_poems(data, column, number_of_poems)
    save_poems = input("Save poems to file? (y/n): ")
    if save_poems == "y":
        save_generated_text(poems)
    elif save_poems == "n":
        pass


def generate_sentences_for_non_default_data(folder):
    number_of_sentences = int(input("Number of sentences to generate: "))

    sentences = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as file:
            for line in file:
                sentences.append(line.strip())
    
    sentences = preprocess_text(sentences)
    generated_sentences = generate_sentences(sentences, number_of_sentences)

    for sentence in generated_sentences:
        print(sentence)

    save_sentences = input("Save poems to file? (y/n): ")
    if save_sentences == "y":
        save_generated_text(generated_sentences)
    elif save_sentences== "n":
        pass

    
def change_training_data_source():
    print("Blank input will revert to default data source.")
    data = input("Enter path for folder containing training data source: ")
    return DEFAULT_DATA if data == "" else data


if __name__ == "__main__":
    main()