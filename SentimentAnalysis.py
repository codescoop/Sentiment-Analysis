import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

text = open("read.txt", encoding="utf-8").read()
lower_case = text.lower()
# print(lower_case)

# print(string.punctuation)
cleaned_text = lower_case.translate(str.maketrans("", "", string.punctuation))
# print(cleaned_text)

tokenized_words = word_tokenize(cleaned_text, "english")
# print(tokenized_words)


## removing stop_words
final_words = []
for word in tokenized_words:
    if word not in stopwords.words("english"):
        final_words.append(word)

# print(final_words)

## Applying NLP Algorithm
emotion_list = []
with open("emotions.txt", "r") as emotion_file:
    for line in emotion_file:
        clear_line = line.replace('\n', " ").replace("'", "").replace(",", "").strip()
        word, emotion = clear_line.split(":")
        # print(word,emotion)

        if word in final_words:
            emotion_list.append(emotion)

# print(emotion_list)

## counting emotions
word_count = Counter(emotion_list)
# print(word_count)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    # print(score)
    neg = score["neg"]
    pos = score["pos"]
    if neg > pos:
        print("Negative Sentiment")
    else:
        print("Positive Sentiment")

sentiment_analyse(cleaned_text)

fig, axl = plt.subplots()
axl.bar(word_count.keys(), word_count.values())
fig.autofmt_xdate()
# plt.bar(word_count.keys(),word_count.values())
plt.savefig("graph.png")
plt.show()