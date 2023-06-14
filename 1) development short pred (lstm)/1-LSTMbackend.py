import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from numpy import hstack
import sqlalchemy
from sklearn.ensemble import IsolationForest
import time


# ---- set database connection
database_username = 'root'
database_password = 'balab'
database_ip       = '172.17.0.7'
database_name     = 'data_hackathon'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format(database_username, database_password,
                                                      database_ip, database_name))


while True :

    data = pd.read_sql('select * from data_source order by TIME_STAMP desc limit 60',con=database_connection)
    data = data.sort_values("TIME_STAMP").reset_index(drop=True)
    data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'])

    baseline = pd.read_csv("baseline_normalization.csv")

    ISO_MODEL  = "isolation_forest.pkl"
    LSTM_MODEL = "model_LSTM.h5"

    def preprocessing(df) :
        
        # FILTER FEATURES
        LIST_FEATURES = ['CURRENT_UV',
                        'CURRENT_SPEED',
                        'SPEED_VG',
                        'SPEED_LW',
                        'REL_WIND_SPEED',
                        'TOTAL_WAVE_HEIGHT',
                        'TOTAL_WAVE_DIRECTION',
                        'TOTAL_WAVE_PERIOD',
                        'WIND_UV',
                        'WIND_SPEED',
                        'PRESSURE_SURFACE',
                        'PRESSURE_MSL',
                        'SEA_SURFACE_SALINITY',
                        'ME1_RPM',
                        'ME1_FOC']
        
        df = df[['TIME_STAMP']+LIST_FEATURES]
        
        # ISOLATION FOREST
        trained_iso = open(ISO_MODEL,'rb')
        iso = pickle.load(trained_iso)
        iso_score = iso.score_samples(df.drop(['TIME_STAMP'],axis=1))
        iso_binary = iso.predict(df.drop(['TIME_STAMP'],axis=1))

        df['ISO_SCORE']  = iso_score
        df['ISO_BINARY'] = iso_binary

        # DAY or NIGHT
        df['DAY_NIGHT']  = df['TIME_STAMP'].apply(lambda x : 1 if 6<=pd.to_datetime(x).hour<=12 else 0)
        df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])
        
        return df

    inputs = preprocessing(data)

    #normalization function
    def norm(x,data):
        xn = (x - min(data)) / (max(data) - min(data))
        return xn
    def inv_norm(xn,data) :
        x = (xn*(max(data) - min(data))) + min(data)
        return x

    # define dataset
    in_seq1 = np.array(norm(inputs['TOTAL_WAVE_HEIGHT'],baseline['TOTAL_WAVE_HEIGHT']))
    in_seq2 = np.array(norm(inputs['CURRENT_UV'],baseline['CURRENT_UV']))
    in_seq3 = np.array(norm(inputs['CURRENT_SPEED'],baseline['CURRENT_SPEED']))
    in_seq4 = np.array(norm(inputs['SPEED_VG'],baseline['SPEED_VG']))
    in_seq5 = np.array(norm(inputs['SPEED_LW'],baseline['SPEED_LW']))
    in_seq6 = np.array(norm(inputs['REL_WIND_SPEED'],baseline['REL_WIND_SPEED']))
    in_seq7 = np.array(norm(inputs['TOTAL_WAVE_DIRECTION'],baseline['TOTAL_WAVE_DIRECTION']))
    in_seq8 = np.array(norm(inputs['TOTAL_WAVE_PERIOD'],baseline['TOTAL_WAVE_PERIOD']))
    in_seq9 = np.array(norm(inputs['WIND_UV'],baseline['WIND_UV']))
    in_seq10 = np.array(norm(inputs['WIND_SPEED'],baseline['WIND_SPEED']))
    in_seq11 = np.array(norm(inputs['PRESSURE_SURFACE'],baseline['PRESSURE_SURFACE']))
    in_seq12 = np.array(norm(inputs['PRESSURE_MSL'],baseline['PRESSURE_MSL']))
    in_seq13 = np.array(norm(inputs['SEA_SURFACE_SALINITY'],baseline['SEA_SURFACE_SALINITY']))
    in_seq14 = np.array(norm(inputs['ME1_RPM'],baseline['ME1_RPM']))
    in_seq15 = np.array(norm(inputs['ME1_FOC'],baseline['ME1_FOC']))
    in_seq16 = np.array(norm(inputs['ISO_SCORE'],baseline['ISO_SCORE']))
    in_seq17 = np.array(norm(inputs['ISO_BINARY'],baseline['ISO_BINARY']))
    in_seq18 = np.array(norm(inputs['DAY_NIGHT'],baseline['DAY_NIGHT']))

    # reshape series
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    in_seq4 = in_seq4.reshape((len(in_seq4), 1))
    in_seq5 = in_seq5.reshape((len(in_seq5), 1))

    in_seq6 = in_seq6.reshape((len(in_seq6), 1))
    in_seq7 = in_seq7.reshape((len(in_seq7), 1))
    in_seq8 = in_seq8.reshape((len(in_seq8), 1))
    in_seq9 = in_seq9.reshape((len(in_seq9), 1))
    in_seq10 = in_seq10.reshape((len(in_seq10), 1))

    in_seq11 = in_seq11.reshape((len(in_seq11), 1))
    in_seq12 = in_seq12.reshape((len(in_seq12), 1))
    in_seq13 = in_seq13.reshape((len(in_seq13), 1))
    in_seq14 = in_seq14.reshape((len(in_seq14), 1))
    in_seq15 = in_seq15.reshape((len(in_seq15), 1))

    in_seq16 = in_seq16.reshape((len(in_seq16), 1))
    in_seq17 = in_seq17.reshape((len(in_seq17), 1))
    in_seq18 = in_seq18.reshape((len(in_seq18), 1))

    # horizontally stack columns
    dataset_input = hstack((in_seq1,in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9, in_seq10, in_seq11, in_seq12, in_seq13, in_seq14, in_seq15, in_seq16, in_seq17, in_seq18))
    dataset_input = dataset_input.reshape(1,60,-1)

    trained_model = tf.keras.models.load_model(LSTM_MODEL)

    predicted = trained_model.predict(dataset_input)
    predicted = inv_norm(predicted,baseline['TOTAL_WAVE_HEIGHT'])

    result = pd.DataFrame({
        'FUTURE_TS' : [data.iloc[-1]['TIME_STAMP'] + pd.Timedelta(minutes=10)],
        'TOTAL_WAVE_HEIGHT' : predicted[0]
    })

    # ---- proses ingestion to databases
    result.to_sql(con=database_connection, name='dm_lstm_prediction', if_exists='replace',index=False)


    for i in tqdm(range(0,60)):
        time.sleep(1)
