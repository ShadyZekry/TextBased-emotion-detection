from empath import Empath
import pandas as pd
import time

#helper function to format time from seconds
def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  return "{0}:{1}:{2}".format(int(hours),int(mins), sec)


depression_categories = ["depressed", "depression", "depressant", "depressing", "depressive",
                     "depressing", "loneliness", "alone", "hostility", "hostile",
                     "hurting", "hurt", "pain", "negative_emotion", "sad", "fear", "stress"]


empath = Empath()


preprocessed_df = pd.read_csv('PreprocessedTweets.csv', usecols=["target", "tokenized_tweet"])

preprocessed_tweets = preprocessed_df[pd.notnull(preprocessed_df["tokenized_tweet"])]["tokenized_tweet"].astype("string")

calc_score_start = time.time()
targets = []
i = 0
for tweet in preprocessed_tweets:
   i += 1
   cat_scores = empath.analyze(tweet, categories=depression_categories)
   total_score = 0
   for cat, score in cat_scores.items():
       total_score += score
   targets.append(total_score)
   if i % 100000 == 0:
        print("Calculating tweet target is " , i / len(preprocessed_tweets) * 100, "% done")


for i in range(len(targets)):
    if targets[i] > 9:
        targets[i] = 9


calc_score_end = time.time()
print("Calculating tweet target elapsed time: ", time_convert(calc_score_end - calc_score_start))

write_csv_start = time.time()

multiclass_dataset = {"target": targets, "tweets":preprocessed_tweets}
multiclass_df = pd.DataFrame(multiclass_dataset)
multiclass_df.to_csv('MulticlassTweets.csv',index=False, columns=["target", "tweets"])

write_csv_end = time.time()
print("Writing csv elapsed time: ", time_convert(write_csv_end - write_csv_start))




