import pandas as pd
import os

def read_data(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise Exception(f'File {file_path} doesn\'t exist')
    else:
        data = pd.read_csv(file_path,
                           usecols=['sofifa_id',
                                    'short_name',
                                    'player_positions',
                                    'overall',
                                    'potential',
                                    'value_eur',
                                    'wage_eur',
                                    'age',
                                    'height_cm',
                                    'weight_kg',
                                    'club_name',
                                    'league_name',
                                    'nationality_name',
                                    'preferred_foot',
                                    'pace',
                                    'shooting',
                                    'passing',
                                    'dribbling',
                                    'defending',
                                    'physic'])
        print('File loaded succesfully')

    data.set_index(['sofifa_id', 'short_name'], inplace=True)
    data['player_positions'] = data['player_positions'].apply(lambda x: x.split(', '))
    return data

