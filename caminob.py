import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import random
from imdb import Cinemagoer
from scipy.sparse import csr_matrix
from final_model import EnsembleRecommender
import pickle
import time


links_df = pd.read_csv("links.csv")
movies_df = pd.read_csv("movies.csv")

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=b83c1c45fe99fda4ebe2d1089882618f&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    overview = data['overview']
    return full_path, overview

def show_movie(movie_id):
    col1,col2 = st.columns(2)
    this_movie = movies_df[movies_df['movieId'] == movie_id]
    print(movie_id,this_movie)
    if (this_movie.empty):
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
    mu_matrix_cp = csr_matrix(mu_matrix.values)
    return mu_matrix, mu_matrix_cp


def callback(titles):
    st.session_state.titles = titles
def show_camino_b():


    st.title("Experimento B")
    st.write("""Disclaimer: Toda la información utilizada en este experimento es anónima y se respeta la confidencialidad de los usuarios. El proposito de este experimento es como parte de una investigación. """)
    st.write("""En este experimento,primero se solicita información personal, luego se pide un rating
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

    edad = st.slider("Edad", 15, 65,25)

    
    st.write("""## Películas rating""")
    st.write(""" Por favor evalue las siguientes Películas con un rating del 1 a 5, donde 5 es el mas postivo. En caso no haya visto la Película, tome en cuenta que tan probable es que la vea segun la infromacion disponible""")
    col1,col2 = st.columns(2)

    first_movie_id = 109487
    this_movie = movies_df[movies_df['movieId'] == first_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==first_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == first_movie_id]["tmdbId"].values[0]
    
    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)
    b_rating = st.slider(" Película 1", 1, 5, 1)

    #-------
    col1,col2 = st.columns(2)

    second_movie_id = 128360
    this_movie = movies_df[movies_df['movieId'] == second_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==second_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == second_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)


    b_rating2 = st.slider(" Película 2", 1, 5, 1)

    #-------
    col1,col2 = st.columns(2)

    third_movie_id = 5618
    this_movie = movies_df[movies_df['movieId'] == third_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==third_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == third_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)

    b_rating3 = st.slider("Película 3", 1, 5, 1)

    #---------
    col1,col2 = st.columns(2)

    fourth_movie_id = 179217
    this_movie = movies_df[movies_df['movieId'] == fourth_movie_id]

    col1.write("**"+"Occupants"+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]== fourth_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == fourth_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)
    
    b_rating4 = st.slider(" Película 4", 1, 5, 1)
    #---------

    col1,col2 = st.columns(2)
    fifth_movie_id = 179585
    this_movie = movies_df[movies_df['movieId'] == fifth_movie_id]

    col1.write("**"+this_movie["title"].values[0]+"**")
    col1.write("Categorias:  "+movies_df[movies_df["movieId"]==fifth_movie_id]['genres'].values[0].replace("|",", "))
    id_tmdb = links_df[links_df["movieId"] == fifth_movie_id]["tmdbId"].values[0]

    image_1 , overview_1 = fetch_poster(id_tmdb)
    col1.write("Trama:  "+str(overview_1))
    col2.image(image_1, width=250)
    
    b_rating5 = st.slider(" Película 5", 1, 5, 1)
    st.write(""" A continuación presione el botón para generar las recomendaciones, puede tardar unos segundos""")
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
  
    ok1 = st.button("Generar recomendaciones ")
    titlesr = []
    

    titles=[]
    this_titles = []
    if ok1:
        with st.spinner('Generando recomendaciones...'):
            sample_df = pd.read_csv("sample_df.csv")

            user_ratings = np.array([b_rating,b_rating2,b_rating3,b_rating4,b_rating5])
            rand_number = np.random.randint(0,9)
            user_id = sample_df["userId"].value_counts().tail(10).index.tolist()[rand_number]
            movies_ids = np.array([first_movie_id,second_movie_id,third_movie_id,fourth_movie_id,fifth_movie_id])
            user_ratings_df = pd.DataFrame({"userId":user_id,"movieId":movies_ids,"rating":user_ratings,"timestamp":time.time()})  
            user_ratings_df = user_ratings_df[["userId","movieId","rating","timestamp"]]

            
        
            
            sample_df = sample_df[sample_df["userId"] != user_id]
            print(sample_df.shape)
            sample_df = sample_df.append(user_ratings_df)
            print(sample_df.shape)
            

           
            
            rating_matrix, rating_matrix_cp = movie_use_matrix_pivot(sample_df)
            filename = 'nmf_model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            item_vector = loaded_model.components_.T
            Ensemble = EnsembleRecommender(sample_df, movies_df,rating_matrix,item_vector)
            print("hey")
            
            titles = Ensemble.Recommend(user_id)

            st.session_state.titles = titles.title


            for i in titles.index:
                show_movie(i)
                this_titles.append(i)


    st.write("Por favor observe las recomendaciones y presione el botón para mostrar las recomendaciones")
    try:
        st.session_state.titles
    except:
        st.session_state.titles = []
    agree = st.checkbox('Mostrar recomendación',on_change=callback(st.session_state.titles))

    if agree:
        st.write(st.session_state.titles)
    rating1 = st.slider("Pelicula1", 1, 5, 1)
    rating2 = st.slider("Pelicula2", 1, 5, 1)
    rating3 = st.slider("Pelicula3", 1, 5, 1)
    rating4 = st.slider("Pelicula4", 1, 5, 1)
    rating5 = st.slider("Pelicula5", 1, 5, 1)

    raitings_s = [rating1,rating2,rating3,rating4,rating5]


    ok = st.button("Generar segundo grupo de recomendaciones ")
    titles = []
    if ok:
        sample_df = pd.read_csv("sample_df.csv")

        user_ratings = np.array(raitings_s)
        rand_number = np.random.randint(0,9)
        user_id = sample_df["userId"].value_counts().tail(10).index.tolist()[rand_number]
        movies_ids = np.array([first_movie_id,second_movie_id,third_movie_id,fourth_movie_id,fifth_movie_id])
        user_ratings_df = pd.DataFrame({"userId":user_id,"movieId":movies_ids,"rating":user_ratings,"timestamp":time.time()})  
        user_ratings_df = user_ratings_df[["userId","movieId","rating","timestamp"]]

        
    
        
        sample_df = sample_df[sample_df["userId"] != user_id]
        print(sample_df.shape)
        sample_df = sample_df.append(user_ratings_df)
        print(sample_df.shape)
        

        
        
        rating_matrix, rating_matrix_cp = movie_use_matrix_pivot(sample_df)
        filename = 'nmf_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        item_vector = loaded_model.components_.T
        Ensemble = EnsembleRecommender(sample_df, movies_df,rating_matrix,item_vector)
        print("hey")
        
        titles = Ensemble.Recommend(user_id)

        
        count = 0
        movies_to_show = []
        for i in range(0,len(raitings_s)):
            if raitings_s[i] >= 4:
                movies_to_show.append( st.session_state.titles.index[i])
                count+=1
            else:
                if st.session_state.titles.index[i] in titles.index:
                    titles.drop(st.session_state.titles.index[i],inplace=True)
        print("count: ",count)
        
        for i in titles.index:
            movies_to_show.append(i)
        movies_to_show = set(movies_to_show)

        for i in movies_to_show:
            show_movie(i)

        st.session_state.title = titles.index
    
    # Generar una nueva pagina

        
                



        
            

    
   
      
    st.markdown("# Evaluación")
   #pasar al camino B  
    st.write("""Una vez revisadas las recomendaciones, por favor complete el siguiente formulario""")
    link = '## [Formulario](https://forms.gle/BCNeQnhf2h1guDZd9)'
    st.markdown(link, unsafe_allow_html=True)
