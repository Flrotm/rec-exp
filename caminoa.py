import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import random
from imdb import Cinemagoer
from scipy.sparse import csr_matrix
from caminob import show_camino_b
from final_model import EnsembleRecommender
import pickle
import time
import os
import warnings

links_df = pd.read_csv("links.csv")
movies_df = pd.read_csv("movies.csv")

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=b83c1c45fe99fda4ebe2d1089882618f&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    if data["poster_path"] is not None:
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        overview = data['overview']
        return full_path, overview
    else:
        return "https://www.movienewz.com/img/films/poster-holder.jpg", "No overview found"



def show_movie(movie_id):
    col1,col2 = st.columns(2)
    fifth_movie_id = movie_id
    this_movie = movies_df[movies_df['movieId'] == movie_id]
    print(this_movie)
    if (len(this_movie["title"].values) == 0):
        this_movie = movies_df[movies_df['title'] == 208038]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)

def movie_use_matrix_pivot(df_):
    mu_matrix = df_.pivot(index = 'userId', 
                          columns = 'movieId', 
                          values = 'rating').fillna(0)
    
    return mu_matrix

def show_camino_a():
    links_df = pd.read_csv("links.csv")
    movies_df = pd.read_csv("movies.csv")

    st.title("Experimento A")
    st.write("""Disclaimer: Toda la información utilizada en este experimento es anónima y se respeta la confidencialidad de los usuarios. El propósito de este experimento es parte de una investigación en ciencia de la computación. """)
    st.write("""En este experimento, primero se solicita información personal, luego se pide un rating
    inicial de un grupo de películas, se muestran las recomendaciones y finalmente se pide una evaluación de estas recomendaciones""")
    st.write("""### Información personal""")

    countries = (
        "Perú",
        "Argentina",
        "Chile",
        "Colombia",
        "Ecuador",
        "España",
        "México",
        "Venezuela",
        "Estados Unidos",
        "Otro"
    )

    education = (
        "Hombre",
        "Mujer",
        "Otro",
    )

    country = st.selectbox("País", countries)
    education = st.selectbox("Genero", education)

    expericence = st.slider("Edad", 15, 65,25)

    
    st.write("""## Rating Inicial""")
    st.write(""" Por favor evalue las siguientes peliculas con un rating del 1 a 5, donde 1 es el mínimo y 5 es el más positivo. En caso no haya visto la película, tome en cuenta qué tan probable es que la vea según la información disponible.""")
    col1,col2 = st.columns(2)

    first_movie_id = 72998
    this_movie = movies_df[movies_df['movieId'] == first_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==first_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == first_movie_id]["tmdbId"].values[0]
    
    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)
    rating = st.slider("Pelicula 1", 1, 5, 1)

    #-------
    col1,col2 = st.columns(2)

    second_movie_id = 202439
    this_movie = movies_df[movies_df['movieId'] == second_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==second_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == second_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)


    rating2 = st.slider("Pelicula 2", 1, 5, 1)

    #-------
    col1,col2 = st.columns(2)

    third_movie_id = 1
    this_movie = movies_df[movies_df['movieId'] == third_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==third_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == third_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)

    rating3 = st.slider("Pelicula 3", 1, 5, 1)

    #---------
    col1,col2 = st.columns(2)

    fourth_movie_id = 103075
    this_movie = movies_df[movies_df['movieId'] == fourth_movie_id]

    col1.write("**"+"The Purge (2013)"+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]== fourth_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == fourth_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)
    
    rating4 = st.slider("Pelicula 4", 1, 5, 1)
    #---------

    col1,col2 = st.columns(2)
    fifth_movie_id = 179595
    this_movie = movies_df[movies_df['movieId'] == fifth_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==fifth_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == fifth_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)

    rating5 = st.slider("Pelicula 5", 1, 5, 1)
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0099ff;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #00ff00;
        color:#ff0000;
        }
    </style>""", unsafe_allow_html=True)
    st.write(""" A continuación presione el botón para generar las recomendaciones.  Esto toma unos segundos""")
    ok = st.button("Generar recomendaciones ")
    if ok:
        with st.spinner('Generando recomendaciones...'):
            warnings.warn("Start")
            sample_df = pd.read_csv("sample_df.csv")

            user_ratings = np.array([rating,rating2,rating3,rating4,rating5])
            rand_number = np.random.randint(0,9)
            user_id = sample_df["userId"].value_counts().tail(10).index.tolist()[rand_number]
            movies_ids = np.array([first_movie_id,second_movie_id,third_movie_id,fourth_movie_id,fifth_movie_id])
            user_ratings_df = pd.DataFrame({"userId":user_id,"movieId":movies_ids,"rating":user_ratings,"timestamp":"1346916814"})  
            user_ratings_df = user_ratings_df[["userId","movieId","rating","timestamp"]]

            
        
            
            sample_df = sample_df[sample_df["userId"] != user_id]
            print(sample_df.shape)
            sample_df = sample_df.append(user_ratings_df)
            print(sample_df.shape)
            

           
            
            rating_matrix, rating_matrix_cp = movie_use_matrix_pivot(sample_df)
            print(type(rating_matrix))
            item_vector=[]
          
            Ensemble = EnsembleRecommender(sample_df, movies_df,rating_matrix,item_vector)
            print("hey")

            titles = Ensemble.Recommend(user_id)
        

            for i in titles.index:
                show_movie(i)

            st.markdown("## Por favor pase a la sección B del experimento en la parte izquierda")


        
            
            
        
       


    
   
      
    
   #pasar al camino B  
   
