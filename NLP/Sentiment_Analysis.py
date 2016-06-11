def review _to_words(raw_review):
    # Import the pandas package, then use the "read_csv" function to read
    # the labeled training data
    import pandas as pd
    # Import re in order to deal with text like replace, lower case, etc
    import re
    # import NLTK in order to use Natural Language Processing Tool Kit
    import nltk
    from nltk.corpus import stopwords  # Import the stop word list

    # 1. Remove HTML
    # import BeatifulSoup4 into your workspace
    from bs4 import BeautifulSoup
    train = pd.read_csv("labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)

    # Initialize the BeautifulSoup4 obkect on a single movie review
    # example1 = BeautifulSoup(train["review"][0], "html.parser")

    # print the raw review and then the output of get_test(), for
    # comparison

    # print(train["review"][0])
    # print(example1.get_text())

    # 2. Remove non-letters
    # Use regular expressions to do a find-and-replace
    letters_only = re.sub(
        "[^a-zA-Z]",  # The patther to search for
        " ",           # The pattern to replace it with)
        example1.get_text())  # The text to search

    # print (letters_only)

    # 3. Convert to lower case
    lower_case = letters_only.lower()  # Conver to lower case
    # split into individual words
    words = lower_case.split()

    # print(lower_case)
    # print(words)

    # nltk.download()  # Download text data sets, including stop words

    # print(stopwords.words("english"))

    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # Remove stop words from "words"
    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # print(words)
    # 6. Join the words back into one string separated by space, and return the
    # the result
    return(" ".join(meaningful_words))
