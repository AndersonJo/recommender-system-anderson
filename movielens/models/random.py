import random
from typing import List

from movielens.models.base import BaseRecommendModel


class RandomRecommendModel(BaseRecommendModel):

    def predict(self, user_ids: List[int]):
        pred_movies = list()
        for user_id in user_ids:
            _pred_movies = list()
            watched_movies = self.user2mv[user_id]

            while len(_pred_movies) < self.n_recommend:
                movie_id = random.choice(self.movie_ids)
                if movie_id not in watched_movies:
                    _pred_movies.append(movie_id)
            pred_movies.append(_pred_movies)

        return pred_movies
