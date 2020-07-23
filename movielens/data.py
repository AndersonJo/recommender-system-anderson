import os
import pickle
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict

import kaggle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(force: bool = False) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, int]]:
    """
    return data
     - train_data <ndarray>: [(userId, movieId, rating), ...]
     - test_data <ndarray>:  [(userId, movieId, rating), ...]
     - movie_popularity dict: {movieId: count, ...}
     - train_user2mv Dict[dict]: {userId: {movieId: rating, ...}, ...}
     - test_user2mv Dict[dict]:  {userId: {movieId: rating, ...}, ...}
    """

    np.random.seed(1)
    download_path = Path(tempfile.gettempdir()) / 'movielens'
    train_data_path = download_path / 'train-20m-data.npy'
    test_data_path = download_path / 'test-20m-data.npy'
    popularity_path = download_path / 'popularity.pkl'
    train_user2mv_path = download_path / 'train-user2mv.pkl'
    test_user2mv_path = download_path / 'test-user2mv.pkl'

    # Load Data
    if not force and popularity_path.exists() and train_user2mv_path.exists() and test_user2mv_path.exists():
        print('Load preprocessed data')
        # train_data = np.load(open(train_data_path, 'rb'))
        # test_data = np.load(open(test_data_path, 'rb'))
        movie_popularity = pickle.load(open(popularity_path, 'rb'))
        train_user2mv = pickle.load(open(train_user2mv_path, 'rb'))
        test_user2mv = pickle.load(open(test_user2mv_path, 'rb'))
        return train_user2mv, test_user2mv, movie_popularity

    # Downlaod movielens from Kaggle
    print('Download & Preprocess Data')
    kaggle.api.dataset_download_files('grouplens/movielens-20m-dataset', download_path)

    # Unzip the dataset
    zipfile_path = download_path / [x for x in os.listdir(download_path) if x.endswith('.zip')][0]
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    csv_files = [x for x in os.listdir(download_path) if x.endswith('.csv')]

    # Load Movie Dataset
    # movies = pd.read_csv(download_path / 'movie.csv', index_col=0)

    # Load Rating Dataset
    data = load_raw_data(download_path)
    # data = data.merge(movies[['title']], left_on='movieId', right_index=True)

    # Get movie popularity (the number of views by users wrt movies)
    movie_popularity = defaultdict(int)
    _popularity = data.groupby('movieId').count()['userId']
    _popularity = _popularity.sort_values(ascending=False)
    movie_popularity.update(_popularity.to_dict())

    # Split to train and test
    train_data, test_data = train_test_split(data.values, test_size=0.3, random_state=1)

    # to dictionary. {user_id: {movie_id: rating, ...}}
    train_user2mv = make_user_based_data(train_data)
    test_user2mv = make_user_based_data(test_data)

    # np.save(open(train_data_path, 'wb'), train_data)
    # np.save(open(test_data_path, 'wb'), test_data)
    pickle.dump(movie_popularity, open(popularity_path, 'wb'))
    pickle.dump(train_user2mv, open(train_user2mv_path, 'wb'))
    pickle.dump(test_user2mv, open(test_user2mv_path, 'wb'))
    return train_user2mv, test_user2mv, movie_popularity


def load_raw_data(download_path) -> pd.DataFrame:
    return pd.read_csv(download_path / 'rating.csv', usecols=[0, 1, 2])


def make_user_based_data(data):
    user2movies = defaultdict(dict)
    for i in range(len(data)):
        user_id, movie_id, rating = data[i]
        user_id, movie_id = int(user_id), int(movie_id)
        user2movies[user_id][movie_id] = rating

    return user2movies
