import numpy as np
# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd

# Import sklearn
from sklearn.feature_extraction.text import CountVectorizer
# Import class to do the Supervised Learning using a Random Forest
# Classification
from sklearn.ensemble import RandomForestClassifier
# import BeatifulSoup4 into your workspace
from bs4 import BeautifulSoup

# Get data from labeledTrainData to python
train = pd.read_csv("labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)


def review_to_words(raw_review):

    # Import re in order to deal with text like replace, lower case, etc
    import re

    # import NLTK in order to use Natural Language Processing Tool Kit
    # import nltk
    from nltk.corpus import stopwords  # Import the stop word list

    # 1. Remove HTML
    # Initialize the BeautifulSoup4 obkect on a single movie review
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    # example1 = BeautifulSoup(train["review"][0], "html.parser")

    # print the raw review and then the output of get_test(), for comparison

    # print(train["review"][0])
    # print(example1.get_text())

    # 2. Remove non-letters
    # Use regular expressions to do a find-and-replace
    letters_only = re.sub(
        "[^a-zA-Z]",  # The patther to search for
        " ",          # The pattern to replace it with)
        review_text)  # The text to search

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

# This line will give us the first comment clean
# clean_review = review_to_words(train["review"][0])
# print(clean_review)

# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize a empty list to hold the clean reviews
clean_train_reviews = []

# loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range(0, num_reviews):
    # Call our function for each one, and add the result to the list of
    # clean review
    if((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

# Print first cleat test for showing off
print(clean_train_reviews[0])

# Now we need to created the smart part, the module that will lean from
# all comments we have imported, in order to do this we can apply a module
# from skilearn called Bag of words

print("Creating Bag of Words...\n")

# Initialize the "CountVectorizer" object, which is skikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=None,
    preprocessor=None,
    stop_words=None,
    max_features=5000,)

# fit_transform() does two functions:
# 1st - it fits the model and learn the vocabulary
# 2nd - it transform our training data into feature vectors. The input to
# fit_transforms should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()


print(vocab)
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)

print('''
    Training the random forest...this might take a while depending of your
    computer power
    ''')

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run`
forest = forest.fit(train_data_features, train["sentiment"])

# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_review = []

print('Cleaning and parsing the test set movie reviews...\n')
for i in range(0, num_reviews):
    if((i + 1) % 1000 == 0):
        print('Review %d of %d\n' % (i + 1, num_reviews))
    clean_review = review_to_words(test['review'][i])
    clean_test_review.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_review)
test_data_features = test_data_features.toarray()

# User the random forest to make sentiment label predictions
# This is coolest part where is test the learned part
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" columnd and a
# "sentiment" column
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

# Use pandas to write the comma separated output file
output.to_csv("bag_of_words_model.csv", index=False, quoting=3)
