import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser
import utils
from utils import date_convert,date_offset





def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files. Variables returned from this function are passed as input to the metric functions.
    This function needs to be completed.
    '''
    
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    
    events = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
    

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    Event count is defined as the number of events recorded for a given patient.
    This function needs to be completed.

    '''
    event_dead=events[events['patient_id'].isin(mortality['patient_id'])]
    event_alive=events[~events['patient_id'].isin(mortality['patient_id'])]
    
    dead = event_dead.groupby(['patient_id']).size()
    alive = event_alive.groupby(['patient_id']).size()
    
    avg_dead_event_count = dead.mean()

    max_dead_event_count = dead.max()

    min_dead_event_count = dead.min()
    
    avg_alive_event_count = alive.mean()

    max_alive_event_count = alive.max()

    min_alive_event_count = alive.min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    This function needs to be completed.
    '''
    event_dead=events[events['patient_id'].isin(mortality['patient_id'])]
    event_alive=events[~events['patient_id'].isin(mortality['patient_id'])]


    alive = event_alive.groupby(['patient_id','timestamp']).size()
    dead=event_dead.groupby(['patient_id','timestamp']).size()
    
    avg_dead_encounter_count = dead.count(0).mean()

    max_dead_encounter_count = dead.count(0).max()

    min_dead_encounter_count = dead.count(0).min()
    
    

    avg_alive_encounter_count = alive.count(0).mean()

    max_alive_encounter_count = alive.count(0).max()

    min_alive_encounter_count = alive.count(0).min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient. 
    This function needs to be completed.
    '''
    #events.timestamp.apply(dateutil.parser.parse)

    event_dead=events[events['patient_id'].isin(mortality['patient_id'])]
    event_alive=events[~events['patient_id'].isin(mortality['patient_id'])]


    #event_dead['timestamp_y']=pd.datetime.strptime(event_dead['timestamp_y'],'%Y-%m-%d')
    
    
    
    dead_grouped=event_dead.sort('timestamp').groupby(['patient_id'])
    dead=dead_grouped.first().reset_index()[['patient_id','timestamp']]
    
    dead['last']=dead_grouped.last().reset_index()['timestamp']
    
    dead['diff']=dead.apply(lambda x: (date_convert(x['last'])-date_convert(x['timestamp'])).days, axis=1)
    

    alive_grouped=event_alive.sort('timestamp').groupby(['patient_id'])
    alive=alive_grouped.first().reset_index()[['patient_id','timestamp']]
    
    alive['last']=alive_grouped.last().reset_index()['timestamp']
    
    
    alive['diff']=alive.apply(lambda x: (date_convert(x['last'])-date_convert(x['timestamp'])).days, axis=1)
    
    
    avg_dead_rec_len = dead['diff'].mean()

    max_dead_rec_len = dead['diff'].max()

    min_dead_rec_len = dead['diff'].min()

    avg_alive_rec_len = alive['diff'].mean()

    max_alive_rec_len = alive['diff'].max()

    min_alive_rec_len = alive['diff'].min()

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DONOT MODIFY THIS FUNCTION. 
    Just update the train_path variable to point to your train data directory.
    '''
    #Modify the filepath to point to the CSV files in train_data
    
    train_path = os.path.join("../data/train/")
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
    



