import os
import string
import re
import nltk
import random as rand
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from WordState import WordState


DEFAULT_DATA = "PoetryFoundationData.csv"
DEFAULT_COLUMN = "Poem"


def main():
    data         : str  = DEFAULT_DATA
    column       : str  = DEFAULT_COLUMN
    keep_running : bool = True
    
    while keep_running:
        print("\n--- Poem Generator ---")
        print("Current training data source: " + data + "\n")

        try:
            if data == DEFAULT_DATA:
                user_input : int = display_menu_choice_for_default_data()
                if user_input == "1":
                    generate_poems_for_default_data(data, column)
                elif user_input == "2":
                    data = change_training_data_source()
                elif user_input == "0":
                    keep_running = False
            else:
                user_input : int = display_menu_for_non_default_data()
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

    Arguments:
    file (string)    : The path to the CSV/text file.
    column (string)  : The name of the column to read from. 
                       If None, reads from the first column.
    category (string): The category to filter by. If None, returns all rows.

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
    Arguments:
    sentences (list): A list of sentences.

    Returns:
    dict: A dictionary of word states (dictionaries) describing
    the probability of a word following another word.

    """

    states : dict = {}

    for sentence in sentences:
        words : list = sentence.split()

        START_OF_SENTENCE = "#"
        previous_word : str = START_OF_SENTENCE

        for word in words:
                if previous_word not in states:
                    states[previous_word] : dict = {}

                if word not in states[previous_word]:
                    states[previous_word][word] : int = 1
                else:
                    states[previous_word][word] += 1

                previous_word = word

        if previous_word not in states:
            states[previous_word] = {}

    return states


def generate_sentences(sentences : list , num_sentences : int):

    """
    Generates sentences based on the training data using the class WordState.

    Arguments:
    sentences (list): A list of sentences used as training data.
    num_sentences (int): The number of sentences to generate.

    Returns:
    list: A list of generated sentences based on the training data.
    """
    states : dict = {"#": WordState()}
    for sentence in sentences:
        words : list = sentence.split()
        START_OF_SENTENCE = "#"
        previous_word = START_OF_SENTENCE
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


def lower_case(text : pd.DataFrame):
    return text.str.lower()


def remove_punctuations(text : pd.DataFrame):
    return text.apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )


def remove_stopwords(text : pd.DataFrame):
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))
    return text.apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in STOPWORDS]
        )
    )


def remove_frequent_words(text : pd.DataFrame, frequent_words_to_remove : int):
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

    return text.apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in FREQUENT_WORDS]
        )
    )


def remove_rare_words(text : pd.DataFrame, rare_words_to_remove : int):
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

    return text.apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in RARE_WORDS]
        )
    )


def remove_emojis(text : pd.DataFrame):
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)

    return text.apply(lambda x: emoji_pattern.sub(r'', x))


def remove_emoticons(text : pd.DataFrame):
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

    emoticon_pattern = re.compile(
        u'(' + u'|'.join(k for k in EMOTICONS) + u')'
    )

    return text.apply(lambda x: emoticon_pattern.sub(r'', x))


def convert_emoticons_to_words(text : pd.DataFrame):
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

    for emoticon in EMOTICONS:
        text = re.sub(
            u'('+emoticon+')', 
            "_".join(EMOTICONS[emoticon].replace(",","").split()), 
            text
        )

    return text


def remove_urls(text : pd.DataFrame):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return text.apply(lambda x: url_pattern.sub(r'', x))


def preprocess_text(
        data: pd.DataFrame, 
        lower_casing               : bool = True, 
        remove_punctuations        : bool = False,
        remove_stopwords           : bool = False, 
        frequent_words_to_remove   : int  = 0,
        rare_words_to_remove       : int  = 0,
        remove_emojis              : bool = False,
        remove_emoticons           : bool = False,
        convert_emoticons_to_words : bool = False,
        remove_urls                : bool = False
):
    """
    Preprocesses text by applying a series of transformations.
    Arguments:
    data (pd.DataFrame): The text to preprocess.
    lower_casing (bool): Whether to convert all text to lowercase.
    remove_punctuations (bool): Whether to remove all punctuations.
    remove_stopwords (bool): Whether to remove all stopwords.
    frequent_words_to_remove (int): Number of most frequent words to remove.
    rare_words_to_remove (int): The number of most rare words to remove.
    remove_emojis (bool): Whether to remove all emojis.
    remove_emoticons (bool): Whether to remove all emoticons.
    convert_emoticons_to_words (bool): Whether to convert emoticons to words.
    remove_urls (bool): Whether to remove all URLs.
    Returns:
    pd.DataFrame: The preprocessed text.
    """
    text = data.copy()

    if lower_casing:
        text = lower_case(text)
    
    if remove_punctuations:
        text = remove_punctuations(text)

    if remove_stopwords:
        text = remove_stopwords(text)

    if frequent_words_to_remove > 0:
        text = remove_frequent_words(text, frequent_words_to_remove)

    if rare_words_to_remove > 0:
        text = remove_rare_words(text, rare_words_to_remove)

    if remove_emojis:
        text = remove_emojis(text)

    if remove_emoticons:
        text = remove_emoticons(text)

    if convert_emoticons_to_words:
        text = convert_emoticons_to_words(text)

    if remove_urls:
        text = remove_urls(text)

    return text


def sentences_from_poems(poems : pd.DataFrame) -> list:
    """
    Splits poems into sentences.
    Arguments:
    poems (pd.DataFrame): A dataframe of poems.
    Returns:
    list: A list containing all the sentences from the poems.
    """
    sentences = []

    for poem in poems:
        sentences.append(poem.split("\n"))
    
    #list of lists to list of strings
    sentences = [sentence for poem in sentences for sentence in poem]

    return sentences


def process_output_poems(
        poems : list, max_lines_per_poem : int, max_words_per_line : int
) -> list:
    
    """
    Processes output poems by adding new lines, removing unwanted characters,
    and shortening poems to a maximum number of lines and words per line.

    Arguments:

    poems (list): A list of poems.
    max_lines_per_poem (int): The maximum number of lines per poem.
    max_words_per_line (int): The maximum number of words per line.

    Returns:
    list: A list of processed poems ready for display.
    """
    
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
        
        #New line when capitalized word is found
        poem = "\n".join(
            [
                line + "\n" 
                if line and line[0].isupper() 
                else line for line in poem.split("\n")
            ]
        )

        new_poems.append(poem)
    
    new_poems = shorten_poems(
        new_poems, max_lines_per_poem, max_words_per_line
    )
    
    return new_poems


def shorten_poems(
        poems : list, max_lines_per_poem : int, max_words_per_line : int
    ) -> list:
    """
    Shortens poems to a maximum number of lines and words per line.
    Arguments:
    poems (list): A list of poems.
    max_lines_per_poem (int): The maximum number of lines per poem.
    max_words_per_line (int): The maximum number of words per line.
    Returns:
    list: A list of shortened poems.
    """
    max_lines_per_poem = (
        float('inf') if max_lines_per_poem == 0 else max_lines_per_poem
    )
    max_words_per_line = (
        float('inf') if max_words_per_line == 0 else max_words_per_line
    )
    new_poems = []
    for poem in poems:
        lines = poem.split("\n")
        if len(lines) > max_lines_per_poem:
            lines = lines[:max_lines_per_poem]
        
        new_lines = []
        for line in lines:
            words = line.split()
            if len(words) > max_words_per_line:
                words = words[:max_words_per_line]
            
            new_lines.append(" ".join(words))
        
        new_poems.append("\n".join(new_lines))
    
    return new_poems


def generate_poems(
        csv             : str, 
        column          : str, 
        amount_of_poems : int, 
        number_of_lines : int,
        number_of_words : int, 
        category        : str 
):
    """
    Prints generated poems.

    Arguments:
    csv (string): The path to the CSV/text file.
    column (string): The name of the column to read from.
    amount_of_poems (int): The number of poems to generate.
    number_of_lines (int): The number of lines per poem.
    number_of_words (int): The number of words per line.
    category (string): The category to filter by. If None, returns all rows.

    Returns:
    list: A list of generated poems.
    """
    data = read_and_parse_text(csv, column, category)
    data = preprocess_text(data)
    data = sentences_from_poems(data)
    
    poems = [] 

    poems = generate_sentences(data, amount_of_poems)
    
    for i in range(len(poems)):
        if poems[i] == "":
            while poems[i] == "":
                poems[i] = generate_sentences(data, 1)[0]

    poems = process_output_poems(poems, number_of_lines, number_of_words)

    for poem in poems:
        print("##############################################################")
        print(poem)
        print("##############################################################")
    
    return poems


def save_generated_text(text : list):
    """
    Saves generated text to a file specified by the user.
    """
    try:
        file = input("Enter file name: ")

        while os.path.exists(file):
            print(
                "File already exists. Are you sure you want to overwrite it?"
            )
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
        

def display_menu_choice_for_default_data():
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


def get_integer_from_user(input_text : str):
    """
    Provides exception handling whengetting an integer from the user.

    Arguments:
    input_text (string): The text to display when asking for input.

    Returns:
    int: The integer input from the user.
    """

    while True:
        try:
            number = int(input(input_text))
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
    return number


def generate_poems_for_default_data(data : str, column : str):

    """
    Generates poems when the default data source is used.
    Specialized function for generating poems 

    Arguments:
    data (string): The path to the CSV/text file.
    column (string): The name of the column to read from.

    """

    number_of_poems = get_integer_from_user("Number of poems to generate: ")
    number_of_lines = get_integer_from_user("Number of lines per poem: ")
    number_of_words = get_integer_from_user("Number of words per line: ")

    common_tags : dict = { 
        "1": "Time", 
        "2": "Love", 
        "3": "Nature", 
        "4": "Social Commentaries", 
        "5": "Mythology & Folklore", 
        "6": "Arts & Sciences", 
        "7": "Living", 
        "8": None
    }

    print("Is there a specific category you want to generate poems from: ")
    for key, value in common_tags.items():
        if value is not None:
            print(f"Press {key} for {value} poems")
        else:
            print(f"Press {key} for poems for all/no specific categories")

    category = input("Enter choice: ")
    category = common_tags[ category if category in common_tags else "8"]

    poems = generate_poems(
        data, 
        column, 
        number_of_poems,
        number_of_lines, 
        number_of_words, 
        category
    )
    save_poems = input("Save poems to file? (y/n): ")

    if save_poems == "y":
        save_generated_text(poems)
    elif save_poems == "n":
        pass


def generate_sentences_for_non_default_data(folder : str):

    """
    Generates sentences when a non-default data source is used.
    Generalized function for generating sentences based on
    folder containing text files provided by the user.

    Arguments:
    folder (string): The path to the folder containing the text files.

    """
    number_of_sentences = (
        get_integer_from_user("Number of sentences to generate: ")
    )

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
    """
    Changes the training data source.
    Returns:
    string: The path to the new training data source.
    """
    
    print("Blank input will revert to default data source.")
    data = input("Enter path for folder containing training data source: ")
    return DEFAULT_DATA if data == "" else data


if __name__ == "__main__":
    main()