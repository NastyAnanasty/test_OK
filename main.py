import pandas as pd
import sys

ALGO_1 = 'alg1'
ALGO_2 = 'alg2'
ALGO_3 = 'alg3'
ALGO_4 = 'alg4'

algo_list = [ALGO_1, ALGO_2, ALGO_3, ALGO_4]


class PredictInterface:

    def __init__(self, path_file):
        self.data = pd.read_csv(path_file, index_col=0)

    def predict(self, artist: str):
        pass


class IndicesPredictor(PredictInterface):

    def predict(self, artist: str):
        normalise_data = pd.read_csv(r'normalise_data.csv', sep=',', index_col=0)
        index_artist = normalise_data.index.tolist().index(artist)
        similar_artists_cosine = self.data.iloc[index_artist].tolist()
        if index_artist in similar_artists_cosine:
            similar_artists_cosine.remove(index_artist)
        print('KNN recommends:', normalise_data.iloc[similar_artists_cosine].index.tolist())


class SimilarMatrixPredictor(PredictInterface):

    def predict(self, artist: str):
        similar_artists = self.data[artist].rename(columns={artist: 'similarity'}).sort_values(
            'similarity', ascending=False).index.tolist()[:5]
        if artist in similar_artists:
            similar_artists.remove(artist)
        print('My recommends:', similar_artists[:4])


def createPredictor(alg_type: str) -> PredictInterface:
    if alg_type == ALGO_1:
        return IndicesPredictor(r'indices_pearson.csv')
    elif alg_type == ALGO_2:
        return IndicesPredictor(r'indices_cosine.csv')
    elif alg_type == ALGO_3:
        return SimilarMatrixPredictor(r'cosine_data.csv')
    elif alg_type == ALGO_4:
        return SimilarMatrixPredictor(r'pearson_corr_data.csv')


if __name__ == '__main__':
    algo = ALGO_1
    try:
        algo = sys.argv[1]
        if algo not in algo_list:
            algo = ALGO_1
            print('Default algorithm')
    except IndexError:
        print('Default algorithm')

    predictor = createPredictor(algo)

    while True:
        input_str = input('Input an artist:')
        input_str = input_str.strip()
        if input_str == 'stop':
            break
        try:
            predictor.predict(input_str)
        except ValueError:
            print('Not found')
