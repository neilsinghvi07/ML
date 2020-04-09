
       

import tweepy
import csv
import pandas as pd

consumer_key = 'pqN5ROaL6jONqzcxstuTotKYk'
consumer_secret = 'e0HL0YhdzsNrYa5Mz8DuGDjDiqvTc6Z9384V71e7t8LIB3cNly'
access_token = '910732003779186688-62gn4lcS3cr2U33z5F8PSF69wheWTPD'
access_token_secret = 'G4x69RFE1rdO9LWFgQUUCbquJzbDVcW9KBykI11wx7dAu'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('ua.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)




for tweet in tweepy.Cursor(api.search,q='#delhi#pollution' ,count=100, lang="en", since="2017-04-03").items():
	print (tweet.created_at, tweet.text)
	csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])