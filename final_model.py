import pandas as pd
import numpy as np

from annoy import AnnoyIndex
import warnings
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import random
import streamlit as st
# Final Recommendation system
# This is the Ensemble method that combines NCF DL model with MF-ANN model.
# The ensemble recommender loads and takes recommendations from two pre-trained model,
#    and make recommendation based on user's profile by feeding the user into different model,
#    or add weights to each recommendation to make the final recommendation

class EnsembleRecommender():
    def __init__(self,rating_df,movie_df, rating_matrix, item_vector):
        # class initializer input: - rating_df  a user rating dataframe, containing 'userIds', 'movieIds', 'rating'
        #                          - movie_df   a movie info dataframe, containing 'movieIds', 'title', 'genre'
        #                          - userId     a single userId that the model is recommending for
        #                          - rating_martrix    a user-movie matrix
        #                          - item_vector       the vector representation of each movie learned by MF
        #
        # initialize the variables for recommendation functions
        self.rating_df = rating_df
        self.movie_df = movie_df
        self.user_ids = rating_df['userId'].unique()
        self.movie_ids = rating_df['movieId'].unique()
        self.user2user_encoded = {x: i for i, x in enumerate(self.user_ids)}
        self.movie2movie_encoded = {x: i for i, x in enumerate(self.movie_ids)}
        self.movie_encoded2movie = {i: x for i, x in enumerate(self.movie_ids)}
        self.rating_matrix = rating_matrix
        self.item_vector = []
        

    def NCF_recommendation(self,userId,top_k=20):
        # make recommendation based on NCF model
        # input: - top_k  the number of recommendations made
        #        - userId     a single userId that the model is recommending for
        # output: a dataframe containing index as 'movieId','prediction','title','genre'
        
        # load the pre-trained NCF model
        model =  tf.keras.models.load_model('rec_model.h5')
        
        # get the encoded userId
        client_encoded = self.user2user_encoded[userId]
        
        # get user rated movies
        movie_watched = self.rating_df[self.rating_df['userId'] == userId]['movieId'].values
        
        # get the movies user have not rated in which the NCF  will recommend 
        movie_poll_encoded = []
        for item in self.movie_ids:
            if not np.isin(item, movie_watched):
                movie_poll_encoded.append(self.movie2movie_encoded[item])
        
        # encode the unrated movies into a dataframe
        movie_poll_encoded = random.sample(movie_poll_encoded, 5000)
        print("len:" , len(movie_poll_encoded))


        d = {'user_encoded': [client_encoded] * len(movie_poll_encoded), 'movie_encoded' : movie_poll_encoded}
        client_df = pd.DataFrame(d)
        
        # use the model to predict user's rating on these movies
        #print(client_df['user_encoded'], client_df['movie_encoded'])
        ratings = model.predict([client_df['user_encoded'], client_df['movie_encoded']])
        
        # sort the movies according to the predicted ratings and take top k
        top_ratings_idx = ratings.flatten().argsort()[-top_k:][::-1]
        top_ratings = ratings[top_ratings_idx].flatten()
        recommend_movieId = [self.movie_encoded2movie.get(movie_poll_encoded[x]) for x in top_ratings_idx]
        
        # format the output for better user experience
        top_movie_rec = pd.DataFrame({'movieId': recommend_movieId, 'prediction': top_ratings}).set_index('movieId')
        top_movie_rec = top_movie_rec.join(self.movie_df.set_index('movieId'))
        
        return top_movie_rec
    
    # make recommendation based on MF-ANN model
    def get_rated_movies(self,userId,threshold=3):    
        # input:  userid, a rating threshold, movies that are rated below threshold
        # will not be counted 
        # output: a list of high-scored movies that are rated by givern user, a list of corresponding ratings
        #
        all_rates = self.rating_df[self.rating_df['userId'] == userId]
        high_rates = all_rates[all_rates['rating'] >= threshold]['rating'].values
        high_rate_movie = all_rates[all_rates['rating'] >= threshold]['movieId'].values
        return high_rate_movie, high_rates

    
    def ann(self, metric, num_trees):
        # Implement Approximate Nearest Neighborhood to find similar items, save it in 'rating.ann' 
        # input: target movie, rating matrix, item_vectors, metric (can be "angular", "euclidean", "manhattan", "hamming")
        #        number of trees(More trees gives higher precision when querying)
        # output: save it in 'rating.ann' 
        #
        # construct a dictionary where movied id contains its vector representation 
        # print("movies_ids",len(self.movie_ids))
        # rating_dictionary = {self.movie_ids[i]: self.item_vector[i] for i in range(19835)}
        # #pd.DataFrame(rating_dictionary).to_csv('rating_dictionary.csv')
        # # ann method
        # f = len(self.item_vector[1])
        # t = AnnoyIndex(f, metric)  # Length of item vector that will be indexed
        # for key in rating_dictionary:
        #     t.add_item(key, rating_dictionary.get(key))
        # t.build(num_trees) # 10 trees
        # t.save('rating.ann')
        print("ann  done")
    


    
    def ANN_recommendation(self,userId, dimension = 14, metric = 'angular',
                           num_tree=10, threshold=4, top_n=10):
        # use the trained ANN model to recommend the nearest movies to user's rated movies
        # input: - dimension,metric,
        #          num_tree,threshold,   learned parameter from ANN cv
        #          top_n   
        # output: a dataframe containing index as 'movieId','title','genre'
        #
        warnings.warn("ANN_recommendation is not implemented yet")

        u = AnnoyIndex(14, metric)
        u.load('rating.ann')
        warnings.warn("Traje ann")
       
        
        # construct the recommendation for the user
        high_rate_movie, rate = self.get_rated_movies(userId,threshold=threshold)
        movielist = []
        distancelist = []
        
        if len(high_rate_movie) >= 1:
            # find neighborhood of each movies in the high rated movie set
            for movieid in high_rate_movie:
                movie, dist = u.get_nns_by_item(movieid, top_n, include_distances=True)
                movielist.extend(movie[1:])
                
                # get the weighted distance based on rating scores
                weighted_dist = (np.array(dist[1:])/rate[np.where(high_rate_movie == movieid)]).tolist()
                distancelist.extend(weighted_dist)  
        else:
            st.write("Por lo menos un rating mayor a 4 es necesario para generar recomendaciones")
                

        # construct a dataframe for final output
        top_movie_rec = self.movie_df.loc[self.movie_df['movieId'].isin(movielist)].set_index('movieId')
        
        return top_movie_rec
    

    
    def Recommend(self, userId):
        # if the user have not rated any movies, recommend the most popular movies
        # if the user have rated 1 - 50 movies, recommend with NCF model only
        # if the user have rated 51 - 150 movies, recommend with both NCF and ANN model, with more weights on NCF model
        # if the user have rated more than 151 movies, recommend with both NCF and ANN model, with equal weights
        # input: - userId     a single userId that the model is recommending for
        # output: the comprehensive recommendation for the specific user
        # 
        return self.ANN_recommendation(userId)[:1]
        #return self.NCF_recommendation(userId)[:2].append(self.ANN_recommendation(userId).sample(3))
        
