import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import gc
import joblib
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import argparse

def get_fifa_rank_score_dataframe(filename = 'raw_data/international_matches.csv'):
    df = pd.read_csv(filename, parse_dates=['date'])
    df['home_team_goalkeeper_score'] = round(df.groupby("home_team")["home_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
    df['away_team_goalkeeper_score'] = round(df.groupby("away_team")["away_team_goalkeeper_score"].transform(lambda x: x.fillna(x.mean())))
    df['home_team_mean_defense_score'] = round(df.groupby('home_team')['home_team_mean_defense_score'].transform(lambda x : x.fillna(x.mean())))
    df['away_team_mean_defense_score'] = round(df.groupby('away_team')['away_team_mean_defense_score'].transform(lambda x : x.fillna(x.mean())))
    df['home_team_mean_offense_score'] = round(df.groupby('home_team')['home_team_mean_offense_score'].transform(lambda x : x.fillna(x.mean())))
    df['away_team_mean_offense_score'] = round(df.groupby('away_team')['away_team_mean_offense_score'].transform(lambda x : x.fillna(x.mean())))
    df['home_team_mean_midfield_score'] = round(df.groupby('home_team')['home_team_mean_midfield_score'].transform(lambda x : x.fillna(x.mean())))
    df['away_team_mean_midfield_score'] = round(df.groupby('away_team')['away_team_mean_midfield_score'].transform(lambda x : x.fillna(x.mean())))
    df.fillna(50,inplace=True)
    df['year'] = pd.DatetimeIndex(df['date']).year
    return df

def get_worldcup_championship_dic(worldCups_file_path = 'raw_data/WorldCups.csv'):
    worldCups_df = pd.read_csv(worldCups_file_path)
    champ_list = list(worldCups_df[worldCups_df['Year']>=1994]['Winner'].values)
    champ_dic = defaultdict(int)
    for country in champ_list:
        champ_dic[country] += 1
    top_four_list = list(worldCups_df[worldCups_df['Year']>=1994]['Winner'].values)+list(worldCups_df[worldCups_df['Year']>=1994]['Runners-Up'].values)\
    +list(worldCups_df[worldCups_df['Year']>=1994]['Third'].values) +list(worldCups_df[worldCups_df['Year']>=1994]['Fourth'].values)
    top_four_dic = defaultdict(int)
    for country in top_four_list:
        top_four_dic[country] += 1
    
    return champ_dic, top_four_dic

def get_matches_dataframe(data_path, filename):

    rankings = pd.read_csv(os.path.join(data_path, filename))
    rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                            'two_year_ago_weighted', 'three_year_ago_weighted']]
    rankings = rankings.replace({"IR Iran": "Iran"})
    rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

    matches = pd.read_csv(os.path.join(data_path,'results.csv'))
    matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
    matches['date'] = pd.to_datetime(matches['date'])

    world_cup = pd.read_csv(os.path.join(data_path,'World Cup 2018 Dataset.csv'))
    world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
    world_cup = world_cup.dropna(how='all')
    world_cup = world_cup.replace({"IRAN": "Iran", 
                                "Costarica": "Costa Rica", 
                                "Porugal": "Portugal", 
                                "Columbia": "Colombia", 
                                "Korea" : "Korea Republic"})
    world_cup = world_cup.set_index('Team')

    # I want to have the ranks for every day 
    rankings = rankings.set_index(['rank_date'])\
                .groupby(['country_full'], group_keys=False)\
                .resample('D').first()\
                .fillna(method='ffill')\
                .reset_index()

    # join the ranks
    matches = matches.merge(rankings, 
                            left_on=['date', 'home_team'], 
                            right_on=['rank_date', 'country_full'])
    matches = matches.merge(rankings, 
                            left_on=['date', 'away_team'], 
                            right_on=['rank_date', 'country_full'], 
                            suffixes=('_home', '_away'))
    matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
    matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2
    matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']
    matches['score_difference'] = matches['home_score'] - matches['away_score']
    matches['is_won'] = matches['score_difference'] > 0 # take draw as lost
    matches['is_stake'] = matches['tournament'] != 'Friendly'

    return matches

def concatenate_fifa_ranking_scores(matches, fifa_ranking_df):
    matches['year'] = pd.DatetimeIndex(matches['date']).year

    home_ability_score = ['home_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score']
    for column_name in home_ability_score:
        gp = fifa_ranking_df.groupby(['home_team', 'year'])[column_name].mean()
        gp_dict = gp.to_dict()
        default_dict_gp_dict = defaultdict(lambda:50, gp_dict)

        matches['home_dict_key'] = list(zip(matches['home_team'], matches['year']))
        matches[column_name] = matches['home_dict_key'].map(default_dict_gp_dict)

    away_ability_score = ['away_team_goalkeeper_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score']

    for column_name in away_ability_score:
        gp = fifa_ranking_df.groupby(['away_team', 'year'])[column_name].mean()
        gp_dict = gp.to_dict()
        default_dict_gp_dict = defaultdict(lambda:50, gp_dict)

        matches['away_dict_key'] = list(zip(matches['away_team'], matches['year']))
        matches[column_name] = matches['away_dict_key'].map(default_dict_gp_dict)
    return matches

def encode_teamname(matches):
    label_encoder = LabelEncoder()

    team_list = list(matches['home_team'].astype(str).values)+list(matches['away_team'].astype(str).values)
    label_encoder = LabelEncoder()
    label_encoder.fit(team_list)
    matches['home_team_encoded'] = label_encoder.transform(list(matches['home_team'].astype(str).values))
    matches['away_team_encoded'] = label_encoder.transform(list(matches['away_team'].astype(str).values))
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    matches['number_of_champ_home'] = matches.home_team.map(champ_dic)
    matches['number_of_champ_away'] = matches.away_team.map(champ_dic)
    matches['number_of_topFour_home'] = matches.home_team.map(top_four_dic)
    matches['number_of_topFour_away'] = matches.away_team.map(top_four_dic)
    matches['home_team_encoded_norm'] = (matches['home_team_encoded'] - matches['home_team_encoded'].min()) / (matches['home_team_encoded'].max() - matches['home_team_encoded'].min()) 
    matches['away_team_encoded_norm'] = (matches['away_team_encoded'] - matches['home_team_encoded'].min()) / (matches['home_team_encoded'].max() - matches['home_team_encoded'].min()) 
    return matches, label_encoder

def concatenate_number_of_championship_stat(champ_dic, top_four_dic):
    matches['number_of_champ_home'] = matches.home_team.map(champ_dic)
    matches['number_of_champ_away'] = matches.away_team.map(champ_dic)
    matches['number_of_topFour_home'] = matches.home_team.map(top_four_dic)
    matches['number_of_topFour_away'] = matches.away_team.map(top_four_dic)
    return matches

def get_training_data(input_feat, output_feat, train_ratio=0.8):
    train = matches[:int(matches.shape[0] * train_ratio)]
    val = matches[int(matches.shape[0] * train_ratio):]
    train_y = train[output_feat]
    val_y = val[output_feat]
    train_x = train[input_feat]
    val_x = val[input_feat]

    return train_x, train_y, val_x, val_y

def get_test_data(input_feat, output_feat, test_match, test_result, fifa_ranking_df, cur_year, data_path):
    test_x = []
    
    rankings = pd.read_csv(os.path.join(data_path,'fifa_ranking.csv'))
    rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                            'two_year_ago_weighted', 'three_year_ago_weighted']]
    rankings = rankings.replace({"IR Iran": "Iran"})
    rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])

    # I want to have the ranks for every day 
    rankings = rankings.set_index(['rank_date'])\
                .groupby(['country_full'], group_keys=False)\
                .resample('D').first()\
                .fillna(method='ffill')\
                .reset_index()
    world_cup = pd.read_csv(os.path.join(data_path,'World Cup 2018 Dataset.csv'))
    world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
    world_cup = world_cup.dropna(how='all')
    world_cup = world_cup.replace({"IRAN": "Iran", 
                                "Costarica": "Costa Rica", 
                                "Porugal": "Portugal", 
                                "Columbia": "Colombia", 
                                "Korea" : "Korea Republic"})
    world_cup = world_cup.set_index('Team')

    world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                        rankings['country_full'].isin(world_cup.index.unique())]
    world_cup_rankings = world_cup_rankings.set_index(['country_full'])

    for i in range(len(test_match)):
        home = test_match.iloc[i, 0]
        away = test_match.iloc[i, 1]
        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        average_rank = (home_rank + opp_rank) / 2
        rank_difference = home_rank - opp_rank
        point_difference = home_points - opp_points
        home_team = home
        test_x.append([average_rank, rank_difference, point_difference])
    test_x = pd.DataFrame(test_x, columns=train_x[['average_rank','rank_difference','point_difference']].columns)
    test_x['home_team_encoded'] = label_encoder.transform(list(test_match['home'].astype(str).values))
    test_x['away_team_encoded'] = label_encoder.transform(list(test_match['away'].astype(str).values))
    test_x['home_team_encoded_norm'] = (test_x['home_team_encoded'] - matches['home_team_encoded'].min()) / (matches['home_team_encoded'].max() - matches['home_team_encoded'].min()) 
    test_x['away_team_encoded_norm'] = (test_x['away_team_encoded'] - matches['home_team_encoded'].min()) / (matches['home_team_encoded'].max() - matches['home_team_encoded'].min()) 
    
    test_x['number_of_champ_home'] = test_match.home.map(champ_dic)
    test_x['number_of_champ_away'] = test_match.away.map(champ_dic)
    test_x['number_of_topFour_home'] = test_match.home.map(top_four_dic)
    test_x['number_of_topFour_away'] = test_match.away.map(top_four_dic)
    
    home_ability_score = ['home_team_goalkeeper_score', 'home_team_mean_defense_score', 'home_team_mean_offense_score', 'home_team_mean_midfield_score']
    for column_name in home_ability_score:
        gp = fifa_ranking_df.groupby(['home_team', 'year'])[column_name].mean()
        gp_dict = gp.to_dict()
        default_dict_gp_dict = defaultdict(lambda:50, gp_dict)

        test_match['home_dict_key'] = list(zip(test_match['home'], [cur_year]*4))
        test_x[column_name] = test_match['home_dict_key'].map(default_dict_gp_dict)
        
    away_ability_score = ['away_team_goalkeeper_score', 'away_team_mean_defense_score', 'away_team_mean_offense_score', 'away_team_mean_midfield_score']
    for column_name in away_ability_score:
        gp = fifa_ranking_df.groupby(['away_team', 'year'])[column_name].mean()
        gp_dict = gp.to_dict()
        default_dict_gp_dict = defaultdict(lambda:50, gp_dict)

        test_match['away_dict_key'] = list(zip(test_match['away'], [cur_year]*4))
        test_x[column_name] = test_match['away_dict_key'].map(default_dict_gp_dict)

    test_y = test_result
    return test_x[input_feat], test_y[output_feat]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data directory', default='./data')
    parser.add_argument('--raw_data_dir', help='data directory', default='./raw_data')
    parser.add_argument('--cur_year', help='test data year', default=2018)
    args = parser.parse_args()

    save_dir = args.data_dir
    raw_data_dir = args.raw_data_dir

    fifa_ranking_df = get_fifa_rank_score_dataframe(filename = os.path.join(raw_data_dir,'international_matches.csv'))

    champ_dic, top_four_dic = get_worldcup_championship_dic(worldCups_file_path = os.path.join(raw_data_dir,'WorldCups.csv'))
    
    matches = get_matches_dataframe(raw_data_dir,'fifa_ranking.csv')

    matches = concatenate_fifa_ranking_scores(matches, fifa_ranking_df)
    
    matches, label_encoder = encode_teamname(matches)
    matches = concatenate_number_of_championship_stat(champ_dic, top_four_dic)

    input_feat = ['average_rank', 'rank_difference', 'point_difference',
              'number_of_champ_home','number_of_champ_away',
              'number_of_topFour_home','number_of_topFour_away',
              'home_team_encoded_norm','away_team_encoded_norm', 
              'home_team_goalkeeper_score', 'away_team_goalkeeper_score',
              'home_team_mean_defense_score', 'home_team_mean_offense_score',
              'home_team_mean_midfield_score', 'away_team_mean_defense_score',
              'away_team_mean_offense_score', 'away_team_mean_midfield_score']

    output_feat = ['home_score', 'away_score']
    cur_year = args.cur_year

    train_x, train_y, val_x, val_y = get_training_data(input_feat, output_feat, train_ratio=0.8)

    if cur_year == 2018:
        test_match = pd.DataFrame(np.array([['Uruguay', 'France'], ['Brazil', 'Belgium'], ['Sweden', 'England'], ['Russia', 'Croatia']]), columns=['home', 'away'])
        test_result = pd.DataFrame(np.array([[0, 2], [1, 2], [0,2], [3, 4]]), columns=['home_score', 'away_score'])
        
        test_x, test_y = get_test_data(input_feat, output_feat, test_match, test_result, fifa_ranking_df, cur_year, data_path=raw_data_dir)
        test_x.to_pickle(os.path.join(save_dir, f'test_{cur_year}_x.pkl'))
        test_y.to_pickle(os.path.join(save_dir, f'test_{cur_year}_y.pkl'))


    train_x.to_pickle(os.path.join(save_dir, 'train_x.pkl'))
    train_y.to_pickle(os.path.join(save_dir, 'train_y.pkl'))
    val_x.to_pickle(os.path.join(save_dir,'val_x.pkl'))
    val_y.to_pickle(os.path.join(save_dir,'val_y.pkl'))