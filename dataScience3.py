import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# There are two types of recommendation system
# 1: Collaborative
# 2: Content Based

# fetch data
data = fetch_movielens(min_rating=4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')

# train model
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):
    n_users, n_items = data['train'].shape

    # generate recommendation for each user we created
    for user_id in user_ids:
        # movies they alread like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # movies our model pick they may like
        scores = model.predict(user_id, np.arange(n_items))
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("Known positives:")
        for x in known_positives[:3]:
            print("%s" % x)
        print("Recommended:")
        for x in top_items[:3]:
            print("%s" % x)


sample_recommendation(model, data, [3, 25, 450])

# find some another recommendation datasets and train in on three different models compare the results and print the best one
#CHALLENGE part 1 of 3 - write your own fetch and format method for a different recommendation
#dataset. Here a good few https://gist.github.com/entaroadun/1653794
#And take a look at the fetch_movielens method to see what it's doing

#CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
#the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

#CHALLENGE part 3 of 3 - Modify this function so that it parses your dataset correctly to retrieve
#the necessary variables (products, songs, tv shows, etc.)
#then print out the recommended results
