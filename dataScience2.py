import tweepy
from textblob import TextBlob
import csv

consumer_key = 'nj3ep0J2QRokyjbMoyilXn2Al'
consumer_sec = 'uVzhykTKvyyrOnQ5mSvdJbxp9ToWSf2E70wcwup4LsaHgdeiQ0'

access_token = '971105810561601536-djvj8csZbPh8oAIZH24v1ESUPCGzXU0'
access_t_sec = 'OZs4qlYWuUHxvCzmQvFMzyz3OL3x8Gfe6xVdO3dqEFmxA'

auth = tweepy.OAuthHandler(consumer_key, consumer_sec)
auth.set_access_token(access_token, access_t_sec)

api = tweepy.API(auth)
public_tweets = api.search('machine learning')

csv = open("dataScience2.csv", "w")
columnTitleRow = "polarity, tweet, reaction\n"
csv.write(columnTitleRow)

for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    polarity = analysis.polarity
    # tweet = '\t'.join([line.strip() for line in tweet.text])
    tweet = tweet.text.replace("\n", " ")
    if analysis.polarity > 0:
        reaction = 'positive'
    elif analysis.polarity < 0:
        reaction = 'negative'
    else:
        reaction = 'neutral'
    row = str(polarity) + ', ' + str(tweet) + ', ' + reaction + '\n'
    csv.write(row)
