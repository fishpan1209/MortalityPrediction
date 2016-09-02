import utils
import os
import pandas as pd
from datetime import datetime
from datetime import timedelta
from utils import date_convert, date_offset
import dateutil.parser
import collections



def read_csv(filepath):
    
    '''
    
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(os.path.join(filepath) + 'events.csv', parse_dates=['timestamp'])
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(os.path.join(filepath) + 'mortality_events.csv', parse_dates=['timestamp'])

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(os.path.join(filepath) + 'event_feature_map.csv')
   

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    

    event_dead=events[events['patient_id'].isin(mortality['patient_id'])]
    event_alive=events[~events['patient_id'].isin(mortality['patient_id'])]
  

    # for dead, indx=dead_date-30

    if event_dead.empty:
        
        events_grouped=events.sort('timestamp').groupby(['patient_id'])
        events=events_grouped.last().reset_index()[['patient_id','timestamp']]
        events['indx_date']=events.apply(lambda x: datetime.strptime(x['timestamp'],'%Y-%m-%d'),axis=1)
        
        indx_date=events[['patient_id','indx_date']]

    
        #dead_grouped=event_dead.sort('timestamp').groupby(['patient_id'])
    else:
        dead=mortality[['patient_id','timestamp']]
        dead['timestamp']=dead.apply(lambda x: (datetime.strptime(x['timestamp'],'%Y-%m-%d')-timedelta(days=30)),axis=1)
        dead.columns=['patient_id','indx_date']
        if event_alive.empty:
            indx_date=dead
        else: 
            alive_grouped=event_alive.sort('timestamp').groupby(['patient_id'])
            alive=alive_grouped.last().reset_index()[['patient_id','timestamp']]
            alive['indx_date']=alive.apply(lambda x: datetime.strptime(x['timestamp'],'%Y-%m-%d'),axis=1)
            alive=alive[['patient_id','indx_date']]
        
            indx_date=pd.concat([alive,dead],ignore_index=True)

    
    #indx_date=indx_date.apply(lambda x: datetime.strptime(x['indx_date'],'%Y-%m-%d %H-%M-%S'),axis=1)
    
    indx_date.to_csv(os.path.join(deliverables_path) + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False,header=True)
    
    #print indx_date
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    """
    print indx_date

    def date_window(x, no_days=2000):
        return x - timedelta(days=no_days)
    
    print indx_date
    indx_date['pre'] = indx_date['indx_date'].apply(date_window)
    print indx_date['pre']
    filtered_events = pd.merge(events, indx_date, on='patient_id', how='outer')
    filtered_events = filtered_events[(filtered_events.timestamp >= filtered_events.pre) & (filtered_events.timestamp <= filtered_events.indx_date)]
    """
    merged=pd.merge(events, indx_date, how='outer',on='patient_id')


    merged['diff']= merged.apply(lambda x: (x['indx_date']-date_convert(x['timestamp'])).days,axis=1)
    filtered=merged[(merged['diff']<=2000) & (merged['diff']>=0)]
    
    filtered_events=filtered[['patient_id','event_id','value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    #print events.index.size,filtered_events.index.size"""
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum to calculate feature value 
    4. Normalize the values obtained above using min-max normalization
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    events=pd.merge(filtered_events_df,feature_map_df,how='outer',on='event_id')
    events=events.dropna(subset=['value'])
    
    lab=events[events['event_id'].str.contains("LAB")]
    nonlab=events[~events['event_id'].isin(lab['event_id'])]
    
    lab=lab.groupby(['patient_id','event_id'])['value'].count().reset_index()
    nonlab=nonlab.groupby(['patient_id','event_id'])['value'].sum().reset_index()
    
   
    
    all_events = pd.concat([lab, nonlab],ignore_index=True)
    all_events=pd.merge(all_events, feature_map_df,how='left',on='event_id')
   

    max_value=all_events.groupby('event_id')['value'].max().reset_index()
    normalized=pd.merge(all_events, max_value, how='outer',on='event_id')
                          
    normalized['value']=normalized['value_x']/normalized['value_y']
    
    aggregated_events=normalized[['patient_id','idx','value']]
    
    
    aggregated_events.columns=['patient_id','feature_id','feature_value']
    
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    patient_features = {}
    
    grouped=aggregated_events.groupby('patient_id',as_index=False)
    for name,group in grouped:
        feature= grouped.get_group(name)[['feature_id','feature_value']]
        feature_dict=zip(feature.feature_id,feature.feature_value)
        patient_features[name]=feature_dict
        
    #print patient_features
    
    all_events=pd.merge(events, mortality, how='outer',on='patient_id')
    all_events['label'].fillna(value=0,inplace=True)
    all_events=all_events[['patient_id','label']]
    mortality=dict(zip(all_events.patient_id,all_events.label))
    #
    #print mortality

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
   '''
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    
    
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    keys=sorted(patient_features.keys())
    
    for key in keys:


        content_string=str(mortality[key])+' '+utils.bag_to_svmlight(sorted(patient_features[key],key=lambda x:x[0]))
        deliverable1.write(content_string)
        content_string=str(int(key))+' '+content_string
        deliverable2.write(content_string)
        if key==keys[-1]:
           deliverable1.write(' \n')
           deliverable2.write(' \n')
        else:
           deliverable1.write('\n')
           deliverable2.write('\n')
    deliverable1.close()
    deliverable2.close()
        
    

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')


if __name__ == "__main__":
    main()




