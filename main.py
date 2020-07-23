from argparse import ArgumentParser, Namespace

from movielens.data import load_dataset
from movielens.models.random import RandomRecommendModel


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('model', type=str, help='random|')
    opt = parser.parse_args()
    opt.model = opt.model.lower().strip()

    assert opt.model in {'random'}, 'unknown model name'
    return opt


def get_model(model_name):
    model = None
    if model_name == 'random':
        model = RandomRecommendModel()

    assert model is not None
    return model


def main():
    # Initialization
    opt = init()

    # Data
    train_data, test_data, movie_popularity = load_dataset()
    train_pct = len(train_data) / (len(train_data) + len(test_data)) * 100
    test_pct = len(test_data) / (len(train_data) + len(test_data)) * 100
    # print(f'train_data       : {train_data.shape} {train_pct:.2f}%')
    # print(f'test_data        : {test_data.shape}  {test_pct:.2f}%')
    print(f'train_data : {len(train_data)} user IDs | {train_pct:.1f}%')
    print(f'test_data  : {len(test_data)} user IDs | {test_pct:.1f}%')
    print(f'movies     : {len(movie_popularity)} movies')

    # Model
    model = get_model(opt.model)

    # Train
    print(f'\nStart Testing for {opt.model} model')
    model.fit(train_data, movie_popularity)

    # Test
    model.test(test_data)


if __name__ == '__main__':
    main()
