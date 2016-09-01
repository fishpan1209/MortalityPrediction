import time

def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files. Variables returned from this function are passed as input to the metric functions.
    This function needs to be completed.
    '''
    events = ''

    mortality = ''

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    Event count is defined as the number of events recorded for a given patient.
    This function needs to be completed.
    '''
    avg_dead_event_count = 0.0

    max_dead_event_count = 0.0

    min_dead_event_count = 0.0

    avg_alive_event_count = 0.0

    max_alive_event_count = 0.0

    min_alive_event_count = 0.0

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    This function needs to be completed.
    '''
    avg_dead_encounter_count = 0.0

    max_dead_encounter_count = 0.0

    min_dead_encounter_count = 0.0 

    avg_alive_encounter_count = 0.0

    max_alive_encounter_count = 0.0

    min_alive_encounter_count = 0.0

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient. 
    This function needs to be completed.
    '''
    avg_dead_rec_len = 0.0

    max_dead_rec_len = 0.0

    min_dead_rec_len = 0.0

    avg_alive_rec_len = 0.0

    max_alive_rec_len = 0.0

    min_alive_rec_len = 0.0

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DONOT MODIFY THIS FUNCTION. 
    Just update the train_path variable to point to your train data directory.
    '''
    #Modify the filepath to point to the CSV files in train_data
    train_path = ''
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



