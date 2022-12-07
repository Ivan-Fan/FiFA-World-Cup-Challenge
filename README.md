# FiFA-World-Cup-Challenge

> Can you predict which team will win FIFA World Cup Qatar 2022?

![Logo FIFA World Cup Qatar 2022 yang berlangsung di Qatar.-skysports.com-](pic/cup.jpg)

*(Source: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022)*

This is the course project of 18797 CMU, named FIFA World Cup Challenge.

Team Members: Fan Yang, Fuyu Tang, Changsheng Su, Chongyi Zheng. Thanks for your time and effort!

## Abstract

FIFA World Cup [1] is the most famous football competition in the world and it will be held from Nov 20, 2022 to Dec 18, 2022 [2]. In this project, we want to predict results of the quarterfinals as well as the exact goals scored by each team using Machine Learning algorithms. Aiming at those goals, we need to leverage massive amounts of data from historical tournaments of each team and ratings of individual players, which is well-suited for machine learning approaches. Specifically, predicting results can be modeled as a classification problem and predicting goals scored by each team can be modeled as a regression problem. To evaluate our method, we will use data from FIFA World Cup 2018 as validation and finally test prediction accuracy on FIFA World Cup 2022 quarterfinals.

## Prerequisites

1. Prepare the prerequisites

   ```python
   pip install -r requirements.txt
   ```

2. The directory structure of this project is shown as below:

   ```python
   .
   ├── README.md
   ├── data
   │   ├── V1
   │   ├── V2
   │   └── raw_data
   ├── data_preparation.py
   ├── models
   ├── models.py
   ├── pic
   ├── requirements.txt
   ├── train.py
   └── utils.py
   ```

## Data

1. This project has collected and preprocessed the soccer datasets from Kaggle, e.g., [FiFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup?resource=download&select=WorldCupMatches.csv), [FIFA Rankings](https://www.kaggle.com/code/agostontorok/soccer-world-cup-2018-winner/data), [Player Score](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset?select=players_22.csv), [International Matches](https://www.kaggle.com/datasets/brenda89/fifa-world-cup-2022) . It is recommended to put them under directory `data/raw_data`.

2. To preprocess the data:

   ```python
   python data_preparation.py
   ```

   Arguments are shown as below:

   | Argument       | Default           | Description             |
   | -------------- | ----------------- | ----------------------- |
   | --data_dir     | './data/V2'       | Data directory to save. |
   | --raw_data_dir | './data/raw_data' | Raw data directory.     |
   | --cur_year     | 2018              | Test data year.         |

P.S. In our V1 experiments, we observed the inbalance issues of feature importance. To alleviate this problem, we bring more relevant features in to our V2 dataset and achieved better results.

## Usage

1. Train and evaluate ML models on WorldCup dataset. For example,

   ```python
   python train.py --model lgb
   ```

   Some important arguments (listed below, more could be found in `train.py`):

   | Argument    | Default   | Description                                      |
   | ----------- | --------- | ------------------------------------------------ |
   | --model     | lgb       | ML models for training and evaluation: e.g., KNN |
   | --data-dir  | 'data/V2' | Directory path for the data set.                 |
   | --model-dir | 'models'  | Directory for saving trained model files         |
   | --cur-year  | 2018      | Test set year.                                   |

## Results

1. Results for regression-based model

   | Model | Scale    | Test MSE on V1 | Test R2   on V1 | Test Acc on V1 | Test MSE on V2 | Test R2  on V2 | Test Acc on V2 |
   | :---: | :------: | :------------: | :-----------: | :------------: | :-----------: | :------------: | :------------: |
   | PR    | Standard |     2.1654     | -1.1983 | 0.5 | 1.4530 | -1.2368 | 0.5 |
   | RF    | None     | 2.2198 | -1.2657 | 0.25 |     1.4877     |    -1.2650     |      0.25      |
   | GB    | None     | 2.1726 | -1.2065 | 0.25 | 1.9732 | -0.9924 | 0.5 |
   | KRR   | Standard | 2.1813 | -1.2546 | 0.5 | 1.9313 | -1.0260 | 0.75 |
   | LGB   | MinMax   | 2.0754 | -1.1583 | 0.5 | 1.3259 | -0.1720 | 0.5 |

2. Results for classification-based model

   | Model | Scale | Test Acc on V1 | Test Win Acc on V1 | Test Acc on V2 | Test Win Acc on V2 |
   | :---: | :---: | :------------: | :----------------: | :------------: | :----------------: |
   |  KNN  | None  |     0.125      |        0.75        |     0.125      |        0.75        |

3. Feature importance analysis

   ![](pic/feat_importance.png)



Enjoy the World Cup 2022!



## References

[1] Nam-Su Kim and Laurence Chalip. Why travel to the fifa world cup? effects of motives, background, interest, and constraints. *Tourism management*, 25(6):695–707, 2004.

[2] Wikipedia. FIFA World Cup — Wikipedia, the free encyclopedia. http://en.wikipedia. org/w/index.php?title=FIFA%20World%20Cup&oldid=1111687599, 2022. [Online; ac- cessed 30-September-2022].

