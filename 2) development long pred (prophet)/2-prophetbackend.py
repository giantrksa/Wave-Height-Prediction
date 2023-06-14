# -*- coding: utf-8 -*-
"""
TOTAL WAVE HEIGHT PREDICTION
PROPHET BACKEND
@author: Mingyu, Agung, Gian

"""

import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.ensemble import IsolationForest
import time
from datetime import datetime
import pytz
from tqdm import tqdm
import fbprophet
import pickle

# ========================================================================================= #
#                           SET DATABASE CONNECTION - LOAD DATA                             #
# ========================================================================================= #

database_username = 'root'
database_password = 'balab'
database_ip       = '172.17.0.7'
database_name     = 'data_hackathon'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format(database_username, database_password,
                                                      database_ip, database_name))


# ========================================================================================= #
#                                 LOOP EVERY 10 MINUTES                                     #
# ========================================================================================= #

while True:
    dt_now = pd.to_datetime(datetime.now(tz=pytz.timezone('Asia/Seoul')))

    print("=====================================================================================")
    print("                 Copyright © BUSINESS ANALYTICS LAB MGA TEAM                         ")
    print("=====================================================================================")
    print()
    print("PROCESSING TIME :",dt_now)

    # -------------- SET THE PROCESS ONLY HAPPENS AT 10 MINUTES  -------------- #
    count_time = dt_now.minute%10
    if count_time ==  0:

        # ----------------------------------------------------------------------------------------- #
        #                         LOAD LAST 10 HOURS DATA FROM DATAMART                             #
        # ----------------------------------------------------------------------------------------- #

        df = pd.read_sql(""" SELECT * FROM data_hackathon.data_source ORDER BY TIME_STAMP DESC LIMIT 60""", 
                    con=database_connection)# get data last 10 hours

        dff_temp = df.sort_values('TIME_STAMP')
        LIST_FEATURES = ['TIME_STAMP',
                        'CURRENT_UV',
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
                        'ME1_FOC'] # set important features

        dff = dff_temp[LIST_FEATURES]

        # ----------------------------------------------------------------------------------------- #
        #                         GENERATE FEATURES FOR PROPHET                                     #
        # ----------------------------------------------------------------------------------------- #
        
        # ISOLATION FOREST
        iso = IsolationForest(contamination=0.1,random_state=4)
        iso.fit(dff.drop(['TIME_STAMP'],axis=1))
        iso_score = iso.score_samples(dff.drop(['TIME_STAMP'],axis=1))
        iso_binary = iso.predict(dff.drop(['TIME_STAMP'],axis=1))

        dff['ISO_SCORE']  = iso_score
        dff['ISO_BINARY'] = iso_binary


        # DAY or NIGHT
        dff['DAY_NIGHT']  = dff['TIME_STAMP'].apply(lambda x : 1 if 6<=pd.to_datetime(x).hour<=12 else 0)
        dff['TIME_STAMP'] = pd.to_datetime(dff['TIME_STAMP'])

        # ----------------------------------------------------------------------------------------- #
        #                                      FBPROPHET                                            #
        # ----------------------------------------------------------------------------------------- #

        # NORMALIZATION FUNCTION
        def norm(x,data):
            xn = (x - min(data)) / (1.5*max(data) - min(data))
            return xn

        def inv_norm(xn,data) :
            x = (xn*(1.5*max(data) - min(data))) + min(data)
            return x

        # STACKING NORMALIZATION DATA
        def stack_norm(df,target):
            timestamp = df['TIME_STAMP']
            category = df[['ISO_BINARY','DAY_NIGHT']]
            X = df.drop(category.columns,axis=1)
            X = X.drop(['TIME_STAMP',target],axis=1)
            
            res = []
            for i in range(len(X.columns)):
                temp =  np.array(norm(X[X.columns[i]],X[X.columns[i]]))
                res.append(temp)

            # HORIZONTALLY STACK DATA
            dataset = pd.DataFrame(res).T
            dataset.columns = X.columns
            
            y = df[target]
            y = pd.DataFrame({'y':np.array(norm(y,y))}) # <-----------TARGET
            
            
            df_out = pd.concat([timestamp,dataset,category,y],axis=1)
            df_out=df_out.rename(columns = {'TIME_STAMP':'ds'})
            
            return df_out

        df_out = stack_norm(dff,'TOTAL_WAVE_HEIGHT')
        df_out['ds'] = df_out['ds'].dt.tz_localize(None)

        # MODEL DEPLOYMENT
        model = fbprophet.Prophet(seasonality_mode='additive')
        model.fit(df_out)

        trend_out = 12 # two hours prediction

        future_df = model.make_future_dataframe(periods=trend_out,freq='10min') 
        forecast_result = model.predict(future_df)


        # ----------------------------------------------------------------------------------------- #
        #                                            DATAMART                                       #
        # ----------------------------------------------------------------------------------------- #

        temp_out = forecast_result.iloc[-trend_out:][['ds','yhat_lower','yhat_upper','yhat']]
        temp_out['yhat_lower'],temp_out['yhat_upper'],temp_out['yhat'] = inv_norm(temp_out['yhat_lower'],dff['TOTAL_WAVE_HEIGHT']),inv_norm(temp_out['yhat_upper'],dff['TOTAL_WAVE_HEIGHT']),inv_norm(temp_out['yhat'],dff['TOTAL_WAVE_HEIGHT'])

        # ----------------------------------------------------------------------------------------- #
        #                                    PICKLE UPDATED MODEL                                   #
        # ----------------------------------------------------------------------------------------- #

        def local_persist(fname, modelname) :
            fname = fname
            pickle.dump(modelname, open(fname, 'wb'))

        local_persist('model/fb_prophet.sav',model)


        # ----------------------------------------------------------------------------------------- #
        #                                            INGESTION                                      #
        # ----------------------------------------------------------------------------------------- #

        # ---- proses ingestion to databases
        temp_out.to_sql(con=database_connection, name='dm_prophet_prediction', if_exists='replace',index=False)
        print("ingestion_datetime :",pd.to_datetime(datetime.now(tz=pytz.timezone('Asia/Seoul'))))
        print()  
        print("MODEL UPDATE DONE!!! :",dt_now)  

    else :
        print("THIS TIME IS "+str(dt_now.minute)+" MINS, PROCESSING CONSTRAINT, ONLY WORKING ON EVERY 10 MINS")
        print()  
        print("NOT UPDATE TIME :",dt_now)  
        
    print()
    print("=====================================================================================")
    print("   Copying in whole or in part is strictly forbidden without prior written approval  ")
    print("=====================================================================================")
    print()
    

    # ------- TIME SLEEP : SET 60 SECONDS  -----------------
    print("WAITING TIME :")
    for i in tqdm(range(0,60)):
        time.sleep(1)
