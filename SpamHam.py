import re
import numpy as np


#P(ham) = no of documents belonging to category ham / total no of documents
#P(spam) = no of documents belonging to category spam / total no of documents

# P(bodyText | spam) = P(word1 | spam) * P(word2 | spam) * …
# P(bodyText | ham) = P(word1 | ham) * P(word2 | ham) * …

#P(word1 | spam) = count of word1 belonging to category spam /
#                  total count of words belonging to category spam.
#P(word1 | ham) = count of word1 belonging to category ham /
#                  total count of words belonging to category ham.

def open_dataset(folder_path='data'):
    labels = list()
    file_paths = list()

    # Open Labels files
    with open(folder_path+'/labels') as files:
        raw_files = files.readlines()
        for item in raw_files:
            pair = item.split(' ')
            labels.append(pair[0])
            file_paths.append(pair[1])

    # Open actual emails
    emails = list()
    for items in file_paths:
        items = items.strip()
        try:
            files = open(folder_path+'/'+items)
            email = files.read()
            email = re.sub(r'[^a-zA-Z ]', '', email)
            email = email.lower()
            emails.append(email)
        except:
            # print('Invalid set:',items)
            pass
    return labels[1:], emails[1:]

def count_words(labels, emails):
    spam_count = dict()
    ham_count = dict()
    total_words = list()
    for index, strings in enumerate(emails):
        bag_of_words = strings.split()
        for words in bag_of_words:
            if labels[index] == 'spam':
                if list(spam_count.keys()).count(words):
                    spam_count[words] += 1
                else:
                    spam_count[words] = 1
            else:
                if list(ham_count.keys()).count(words):
                    ham_count[words] += 1
                else:
                    ham_count[words] = 1
            if total_words.count(words) == 0:
                total_words.append(words)
        if index % 20 == 0:
            print('Index {0} of {1}'.format(index, len(labels)))

    total_words.sort()
    return spam_count, ham_count, total_words

def count_classes(labels):
    spam_count = int()
    ham_count = int()
    for index, text in enumerate(labels):
        text = text.strip()
        if text == 'spam':
            spam_count += 1
        elif text == 'ham':
            ham_count += 1
    return spam_count, ham_count

def calculate_category(text, p_word_spam, p_word_ham, Lambda=0):
    spam_prob = 1.0
    ham_prob = 1.0
    text = text.split()

    for words in text:
        spam_prob *= (list(p_word_spam.keys()).count(words) + Lambda) / len(p_word_spam)
        ham_prob *= (list(p_word_ham.keys()).count(words) + Lambda) /len(p_word_ham)

    return [spam_prob, ham_prob]


if __name__ == '__main__':
    labels, emails = open_dataset('data')
    p_ham_count, p_spam_count = count_classes(labels)

    # Marginal Probabilities of spam and ham class
    p_ham = p_ham_count/len(labels)
    p_spam = p_spam_count/len(labels)

    # Words in spam, ham and unique words.
    p_word_spam, p_word_ham, V = count_words(labels, emails)

    # Conditional probabilities
    #  Replace with testing data
    Lambda = 1
    p_words_category = calculate_category(emails[-1], p_word_spam, p_word_ham, Lambda)

    # Classify
    ham_prob = p_ham * p_words_category[1]
    spam_prob = p_spam * p_words_category[0]

    print(ham_prob > spam_prob, labels[-1])
