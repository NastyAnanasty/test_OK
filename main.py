import pandas as pd
import sys

ALGO_1 = 'alg1'
ALGO_2 = 'alg2'
ALGO_3 = 'alg3'
ALGO_4 = 'alg4'

algo_list = [ALGO_1, ALGO_2, ALGO_3, ALGO_4]


class PredictInterface:

    def __init__(self, path_file):
        self.data = pd.read_parquet(path_file, engine='pyarrow')

    def predict(self, artist: str):
        pass


class ModifiedPredictor(PredictInterface):

    def get_user_listen_artist(self, normalise_data, user_id, similar_users) -> pd.DataFrame:
        pass

    def get_for_loop(self, user_id: str) -> pd.DataFrame:
        pass

    @staticmethod
    def find_user_like_artist(artist: str):
        join_data = pd.read_parquet(r'update_data/join_data.parquet', engine='pyarrow')
        max_scrob = join_data[join_data['artist_name'] == artist]['scrobbles'].max()
        user_id = join_data[(join_data['scrobbles'] == max_scrob) & (join_data['artist_name'] == artist)][
            'user_id'].values
        user_id = user_id.astype('str')
        return user_id

    def similar_user_predict(self, user_id: str, similar_users, artist: str):
        normalise_data = pd.read_parquet(r'update_data/normalise_data.parquet', engine='pyarrow')
        users_listen_artists = self.get_user_listen_artist(normalise_data, user_id, similar_users)
        item_score = {}
        similar_users = self.get_for_loop(user_id)
        for i in users_listen_artists.columns:
            rating = users_listen_artists[i]
            total = 0
            count = 0
            for u in similar_users.index:
                if pd.isna(rating[u]) == False:
                    score = similar_users.loc[u].values.astype('float') * rating[u]
                    total += score
                    count += 1
            item_score[i] = total / count
        item_score = pd.DataFrame(item_score.items(), columns=['artist', 'scrobble_score'])
        ranked_item_score = item_score.sort_values(by='scrobble_score', ascending=False)
        if artist in ranked_item_score:
            ranked_item_score.remove(artist)
        return ranked_item_score


class IndicesPredictor(ModifiedPredictor):

    def get_user_listen_artist(self, normalise_data, user_id, similar_users):
        user_id_listen_artists = normalise_data.loc[user_id].dropna(axis=1, how='all')
        users_listen_artists = normalise_data[normalise_data.index.isin(similar_users)].dropna(axis=1, how='all')
        users_listen_artists.drop(user_id_listen_artists, axis=1, inplace=True, errors='ignore')
        return users_listen_artists

    def get_users_distance(self, user_id: str):
        self.data.index = self.data.index.astype('str')
        similar_users = self.data['indices'].loc[user_id].tolist()
        similar_users = [str(x) for x in similar_users[0]]
        distance_similar_users = self.data['distances'].loc[user_id].tolist()
        distance_similar_users = [str(x) for x in distance_similar_users[0]]
        similar_users.remove(user_id)
        distance_similar_users.remove(distance_similar_users[0])
        data_distance = pd.DataFrame(data=distance_similar_users, index=similar_users)
        return similar_users, data_distance

    def get_for_loop(self, user_id: str):
        _, data_distance = self.get_users_distance(user_id)
        return data_distance

    def predict(self, artist: str):
        user_id = self.find_user_like_artist(artist)
        similar_users, data_distance = self.get_users_distance(user_id)
        ranked_item_score = self.similar_user_predict(user_id, similar_users, artist)
        print('KNN recommends:', list(ranked_item_score['artist'].iloc[0:5].values))


class SimilarMatrixPredictor(ModifiedPredictor):

    def get_user_listen_artist(self, normalise_data, user_id, similar_users):
        user_id_listen_artists = normalise_data.loc[user_id].dropna(axis=1, how='all')
        users_listen_artists = normalise_data[normalise_data.index.isin(similar_users.index)].dropna(axis=1, how='all')
        users_listen_artists.drop(user_id_listen_artists.columns, axis=1, inplace=True, errors='ignore')
        return users_listen_artists

    def get_similar_users(self, user_id: str):
        data_copy = self.data
        if user_id[0] in data_copy.index:
            data_copy.drop(index=user_id, inplace=True)
        user_corr = data_copy[user_id]
        similar_users = user_corr.sort_values(by=[user_corr.columns[0]], ascending=False)[:10]
        return similar_users

    def get_for_loop(self, user_id: str):
        similar_users = self.get_similar_users(user_id)
        return similar_users

    def predict(self, artist: str):
        user_id = self.find_user_like_artist(artist)
        similar_users = self.get_similar_users(user_id)
        ranked_item_score = self.similar_user_predict(user_id, similar_users, artist)
        print('My recommends:', list(ranked_item_score['artist'].iloc[0:5].values))


def createPredictor(alg_type: str) -> PredictInterface:
    if alg_type == ALGO_1:
        return IndicesPredictor(r'update_data/indices_pearson.parquet')
    elif alg_type == ALGO_2:
        return IndicesPredictor(r'update_data/indices_cosine.parquet')
    elif alg_type == ALGO_3:
        return SimilarMatrixPredictor(r'update_data/cosine_data.parquet')
    elif alg_type == ALGO_4:
        return SimilarMatrixPredictor(r'update_data/pearson_corr_data.parquet')


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
        except IndexError:
            print('Not found')
