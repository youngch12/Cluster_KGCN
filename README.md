# Cluster-KGCN


> Cluster Knowledge Graph Convolutional Networks for Recommender Systems



1) Download metis-5.1.0.tar.gz from http://glaros.dtc.umn.edu/gkhome/metis/metis/download and unpack it
2) cd metis-5.1.0
3) make config shared=1 prefix=~/.local/
4) make install
5) export METIS_DLL=~/.local/lib/libmetis.so
<!--5) export METIS_DLL=~/.local/lib/libmetis.dylib-->



### Files in the folder

- `data/`
  - `movie-20m/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie-1m/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
- `src/`: implementations of Cluster-KGCN.




### Running the code
- Movie-20m
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  Download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ mv ml-20m/ratings.csv data/movie/
  $ cd src
  $ python3 preprocess.py -d movie-20m
  ```
  - open `src/main.py` file;

  - comment the code blocks of parameter settings for MovieLens-20M;

  - uncomment the code blocks of parameter settings for movie-1m;

  - ```
    $ python3 main.py
  
  
- Movie-1m
  - ```
     - ```
    $ cd src
    $ python3 preprocess.py -d movie-1m
    ```
  - open `src/main.py` file;

  - comment the code blocks of parameter settings for MovieLens-20M;

  - uncomment the code blocks of parameter settings for movie-1m;

  - ```
    $ python3 main.py $ cd src
    $ python3 preprocess.py -d movie-1m
    ```
  - open `src/main.py` file;

  - comment the code blocks of parameter settings for MovieLens-20M;

  - uncomment the code blocks of parameter settings for movie-1m;

  - ```
    $ python3 main.py
 
   
   
- Music
  - ```
    $ cd src
    $ python3 preprocess.py -d music
    ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for music;
    
  - ```
    $ python3 main.py
    ```

<!-- tensorboard --logdir movie_output/-->