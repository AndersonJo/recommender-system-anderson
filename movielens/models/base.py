from abc import abstractmethod, ABC
from datetime import datetime
from typing import Dict, Union, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseRecommendModel(BaseEstimator, ClassifierMixin, ABC):

    def __init__(self, n_recommend=10):
        self.n_recommend = n_recommend
        self.user2mv: Union[Dict[int, dict], None] = None
        self.movie_popularity: Union[dict, None] = None
        self.movie_ids = list()
        self.n_movies = 0

    @abstractmethod
    def predict(self, user_ids: List[int]):
        raise NotImplementedError

    def fit(self, user2mv: Dict[int, dict], movie_popularity: dict):
        self.user2mv = user2mv
        self.movie_popularity = movie_popularity
        self.movie_ids = list(movie_popularity.keys())
        self.n_movies = len(movie_popularity)

    def test(self, user2mv: Dict[int, Dict[int, int]], seed: int = 0):
        np.random.seed(seed)
        start_dt = datetime.now()
        users = list(user2mv.keys())
        pred_rcmd_movies = self.predict(users)
        all_rcmd_movies = set()

        n_correct = 0  # used for TP + FP
        n_test_movies = 0  # used for TP + FN
        n_rcmd = 0
        popularity_sum = 0
        for i, user_id in enumerate(users):
            test_movies = set(user2mv[user_id])  # the test user will watch the movie.
            rcmd_movies = set(pred_rcmd_movies[i])  # recommended movies (dict)
            all_rcmd_movies |= rcmd_movies

            corrected_movies = test_movies & rcmd_movies
            n_correct += len(corrected_movies)
            n_test_movies += len(test_movies)
            n_rcmd += self.n_recommend
            popularity_sum += sum([np.log1p(self.movie_popularity[mv_id]) for mv_id in corrected_movies])

        precision = n_correct / n_rcmd  # precision = TP/(TP+FP)
        recall = n_correct / n_test_movies  # recall = TP/(TP+FN)
        f1_score = 2 * precision * recall / (precision + recall)
        coverage = len(all_rcmd_movies) / self.n_movies  # unique_추천된_영화_갯수/전체_영화_갯수
        popularity = popularity_sum / n_rcmd
        time = datetime.now() - start_dt

        print(f'precision:{precision:.4f} | recall:{recall:.4f} | f1:{f1_score:.4f} | coverage:{coverage:.2f} | '
              f'popularity:{popularity:.2f} | time:{time}')
