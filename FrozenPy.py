# =========================
#  Necessary packages
# =========================

import numpy as np
import pandas as pd
import os

# =========================
#  Loading data
# =========================

def read_out(DIR_NAME,prefix=''):
    
    """Reads all MedPC .out files from the input directory, reformats the data input simple columns with specified 
    ID header, and saves as .raw.csv file in the same directory.
    
    Note: This file does not resave files if they already exist. To rewrite files the existing file must be deleted.
    
    Parameters
    ----------
    DIR_NAM : str
        directory containing .out files 
    prefix : str, optional
        desired ID prefix. (ex. 'MT' for MT1, MT2, MT3, etc.)
        
    Returns
    -------
    
    This function does not return variables, but instead saves the read data into a new .raw.csv file.
    """
        
    # Read .out files from pre-specified directory
    os.chdir(DIR_NAME)
    for file in os.listdir(DIR_NAME):
        if (file.endswith('.out')):

            # Read row with rat names from .csv
            Names = pd.read_csv(os.path.join(DIR_NAME, file), # File to open
                                engine   = 'python',
                                skiprows = 10,                # Skip rows before Names *make 8 if adjusted .out file*
                                nrows = 1,                    # Read only Names row
                                header= None,                 # No header
                                sep = 'Subject ID ',          # Delimiter
                                skipinitialspace = True)
            Names = _format_names(Names, prefix) # format names

            # Create Dataframe .csv data
            df = pd.read_csv(os.path.join(DIR_NAME, file),
                             delimiter='\t',
                             skiprows = 14,
                             names=Names)
            df.reset_index(drop=True,inplace=True)                 # Reset indexing
            df.rename(columns=lambda x: x.strip(),inplace=True)    # remove whitespace

            # add a row of zeros to replace first skipped row
            new_row = np.zeros([1, len(df.columns)])
            new_row = pd.DataFrame(new_row, columns=df.columns)
            df = pd.concat([new_row, df])

            # Write new .csv if it does not exist
            NewCSVname = os.path.splitext(file)[0] + '.raw.csv'                 # name for new csv file
            if not os.path.isfile(NewCSVname):                                  # if file does not exist,
                df.to_csv(NewCSVname, index=False)                              # save
                print('Saved ' + NewCSVname +'!')
            else:                                                               # else,
                print(NewCSVname + ' already exists. Delete to rewrite file.')  # don't save

    print('Done!')
    
def _format_names(Names,prefix=''):
    Names = Names.drop(0,axis=1)                # Drop first blank column
    Names = Names.values.ravel()                # Remove brackets from numbers

    if Names.dtype.type is np.int_:
            Names = np.char.mod('%d', Names)    # int -> char
            Names = [prefix + i for i in Names] # add prefix

    return Names


def read_rawcsv(fname, delete='Y'):
    
    """Reads and creates a pandas dataframe from a specified .raw.csv file created by read_out. 
    Can also delete specified columns so that only desired data is loaded.
    
    Parameters
    ----------
    fname : str
        name of the desired file 
    delete : str, optional
        deletes columns labled 'DELETE' in the .raw.csv file. 'y' for yes, 'n' for no. ('y' by default)
        
    Returns
    -------
    rawcsv_df : pandas dataframe
        dataframe containing .raw.csv data and corresponding IDs

    """

    rawcsv_df = pd.read_csv(fname, sep=',')                              # read .raw.csv

    if delete.lower() in ['y', 'yes']:                                   # if 'y' or 'yes',
        rawcsv_df.drop(rawcsv_df.filter(regex='DELETE'),1,inplace=True)  # remove 'DELETE' columns
        print('Dropped "DELETE" columns in ' + fname)
    else:                                                                # else,
        print('Did not drop "DELETE" columns in ' + fname)               # don't delete

    return rawcsv_df


def read_metadata(metadata_dir, sep=','):

    """Find and load .metadata.csv files from a given directory
    
    Parameters
    ----------
    metadata_dir : str
        the directory where the .metadata.csv file is located
    sep : str, optional
            the seperator used to read the csv file (default: ',')
        
    Returns
    -------
    metadata : pandas dataframe
        dataframe containing the respective metadata
    """ 
    
    os.chdir(metadata_dir)                                                  # change directory
    for fname in os.listdir(metadata_dir):                                  # search directory
        if (fname.endswith('.metadata.csv')):                               # find metadata
            metadata = pd.read_csv(os.path.join(metadata_dir, fname), sep)  # open metadata
            print('Metadata from ' +fname+ ' successfully loaded')
    return metadata

# =====================
#  Detecting freezing
# =====================
def detect_freezing(Threshold_df, threshold = 10):
    
    """Detects freezing using a threshold from raw motion (MedPC Threshold) data. This function
    loops through the array one row at a time and determines if the values are < 'threshold' for
    at least one second. A new array of 1s and 0s are created denoting freezing and not freezing,
    respectively. 
    
    Note: This code will need to be updated with inputs defining Fs and freezing (ex. immobile for 
    < 1 sec or more/less) to be broadly useful outside the Maren Lab. 

    Parameters
    ----------
    Threshold_df : pandas dataframe
        dataframe generated from readRaw()
    threshold : int, optional
        desired threshold, 10 by default
        
    Returns
    -------
    Behav_df : pandas dataframe
        dataframe containing both freezing (key: 'Freezing') and raw motion (key: 'Threshold') data.
        Freezing data is an array of 1s and 0s denoting freezing and not-freezing.
    """
    
    Threshold_values = Threshold_df.values                   # pd dataframe to np array
    rows = np.size(Threshold_values,0)                       # n rows
    columns = np.size(Threshold_values,1)                    # n columns

    Freezing_np = np.zeros((rows,columns))                   # initialize
    for c in range(columns):                               # loop through columns
        for r in range(5,rows):                            # loop through rows
            data2check = Threshold_values[r-5:r,c]           # 1 second of data to check

            if all(n <= threshold  for n in data2check):     # if freezing => 1
                Freezing_np[r,c] = 1

    Freezing_df = pd.DataFrame(Freezing_np)                  # np array to pd dataframe
    #Freezing_df = Freezing_df.multiply(100)                    # multiple by 100
    Corrected_Freezing = correct_freezing(Freezing_df)
    
    Corrected_Freezing.columns = Threshold_df.columns    # add column names from input df
    Corrected_Freezing = Corrected_Freezing.multiply(100)
    
    Behav_df = pd.concat([Threshold_df, Corrected_Freezing],keys=['Threshold','Freezing'])

    return Behav_df

def correct_freezing(Freezing_df):

    """Corrects freezing detected by detectFreezing(), which only defines freezing one sample at a time,
    and thus cannot account for freezing onset in which 1 second of freezing must be counted. For example,
    at freezing onset 5 samples (1 sec) must be below below threhsold values, but only detectFreezing()
    only defines 0.2 sec of freezing at time. So, this function looks for freezing onset and changes that 
    '1' to a '5' to account for this.
    
    Note: This code will also need to be updated with Fs to be used outside the Maren Lab.

    Parameters
    ----------
    Freezing_df : pandas dataframe
        dataframe generated from detectFreezing()
        
    Returns
    -------
    Corrected_Freezing_df : pandas dataframe
        dataframe containing 0s, 1s, and 5s denoting not-freezing, freezing, and freezing-onset.
    """    
    
    # prep data
    Freezing_values = Freezing_df.values              # pd dataframe to np array
    rows = np.size(Freezing_values,0)                 # n rows
    columns = np.size(Freezing_values,1)              # n columns

    # correct freezing
    Freezing_final = np.zeros((rows,columns))         # initialize
    for c in range(0,columns):                        # loop through columns

        for r in range(4,rows):                     # loop through rows
            previous = Freezing_values[r-1,c]         # previous row
            current = Freezing_values[r,c]            # current row

            if current == 0:                          # if 0, keep as 0
                Freezing_final[r,c] = 0
            elif current == 1 and previous == 0:      # if first 1, correct to 5
                Freezing_final[r,c] = 5
            elif current == 1 and previous == 1:      # if not first 1, keep as 1
                Freezing_final[r,c] = 1

    Corrected_Freezing_df = pd.DataFrame(Freezing_final)           # np array to pd dataframe

    return Corrected_Freezing_df

# ============================
#  Slicing / Formating output
# ============================

def slicedata(df, n_trials, start_time, length, ITI, fs=5, Behav='Freezing'):
    
    """Gets timestamps then slices and averages data accordingly

    Parameters
    ----------
    df : pandas dataframe
        dataframe generated from Threshold2Freezing() 
    n_trials : int, optional
        number of trials
    start_time : int, optional
        time of first onset of specified stimulus (CS, US, ISI, etc.)
    length : int, optional
        length in seconds of specified stimulus
    ITI : int, optional
        lenth in seconds of ITI
    fs : int, optional
        sampling frequency (default=5 Hz)
    Behav : str
        desired behavioral data ('Freezing' or 'Threshold'; default='Freezing')
        
    Returns
    -------
    final_data
        a pandas dataframe of averaged data slices from the specified stimulus
    """
    
    # TODO: Need to make this its own function. Would be useful for other things
    
    # get timestamps
    timestamps = np.zeros([n_trials,2],dtype=int)                                # initialize
    for trial in range(0,n_trials):                                              # loop through trials
        timestamps[trial]  = [start_time+ITI*trial, start_time+length+ITI*trial] # start, stop timestamps

    # slice data with timestamps and average      
    final_data = np.array([])                                         # initialize
    for (start, stop) in timestamps:                                  # loop through timestamps
        averaged_trial = df.xs(Behav)[start*fs+1:stop*fs+1].mean().values # slice and average
        final_data  = np.append(final_data, averaged_trial)           # append
        
    return final_data

def get_averagedslices(df,Trials,BL=180,CS=10,US=2,ISI=58,fs=5,Behav='Freezing',Group=[]):
    
    """Slices and averages data for baseline and individual stimuli within trials

    Parameters
    ----------
    df : pandas dataframe
        dataframe generated from Threshold2Freezing() 
    BL : int, optional
        length of baseline period in seconds
    CS : int, optional
        lengths of CS in seconds
    US : int, optional
        length in seconds of US
    ISI : int, optional
        lenth in seconds of ISI
    Trials : int, optional
        numbers of trials
    fs : int, optional
        sampling frequency (default=5 Hz)
    Behav : str
        desired behavioral data ('Freezing' or 'Threshold'; default='Freezing')
    Group: str
        group metadata to assign. mainly useful for within-subjects data where the same subjects
        have different experimental conditions. Leave as default if not within-subjects. 

    Returns
    -------
    BL_df
        a pandas dataframe with the averaged baseline data
    Trials_df
        a pandas dataframe with averaged CS, US, and ISI data
    """
        
    # Baseline 
    ID = df.xs(Behav).columns                                                   # get IDs
    
    BL_timestamps = [0,BL*fs]                                                   # BL timestamps
    BL_data = df.xs(Behav)[BL_timestamps[0]:BL_timestamps[1]].mean().values     # slice and average data
    
    dict4pandas = {'ID': ID, 'BL': BL_data}                                     # BL dataframe
    BL_df   = pd.DataFrame(dict4pandas)                                     

    
    # Trial prep
    ID_metadata = np.tile(ID,Trials)                                         # ID metadata
    Trial_metadata = [ele for ele in range(1,Trials+1) for i in range(len(ID))] # trial metadata length of n_rats
    ITI = CS+US+ISI                                                  # ITI length

    # CS
    CS_data = slicedata(df, n_trials=Trials, start_time=BL,                     # slice data
                        length=CS, ITI=ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'CS': CS_data}   # CS dataframe
    CS_df = pd.DataFrame(dict4pandas)


    #  US
    start_time = BL + CS                                          # start time
    US_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length=US, ITI=ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'US': US_data}   # US dataframe
    US_df = pd.DataFrame(dict4pandas)


    #  ISI
    start_time = BL + CS + US                                                   # start time
    ISI_data = slicedata(df, n_trials=Trials, start_time=start_time,            # slice data
                         length = ISI, ITI = ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'ISI': ISI_data} # ISI dataframe
    ISI_df   = pd.DataFrame(dict4pandas)
    
    # Make Trials df
    Trials_df = pd.merge(CS_df, US_df, on=['ID', 'Trial'], copy='True')         # combine CS and US data
    Trials_df = pd.merge(Trials_df, ISI_df, on=['ID', 'Trial'], copy='True')    # add ISI data
    
    # Add Group metadata, if any
    if any(Group):
        
        Group_metadata   = [ele for ele in [Group] for i in range(len(ID))]     # group metadata
        
        dict4pandas = {'ID': ID, 'Group': Group_metadata}                       # group dataframe
        Group_df    = pd.DataFrame(dict4pandas)
        
        # merge group df to others
        BL_df      = pd.merge(Group_df,BL_df,on='ID',copy='True')               # BL + group
        Trials_df  = pd.merge(Group_df,Trials_df,on='ID',copy='True')           # Trials + group
        
    return BL_df, Trials_df



def get_averagedslices_flight(df,BL,SCS,US,ISI,Trials,fs=5,Behav='Freezing',Group=[]):
    
    """Flight version of get_averagedslices(). Slices and averages data for 
       baseline and individual stimuli within trials.

    Parameters
    ----------
    df : pandas dataframe
        dataframe generated from Threshold2Freezing() 
    BL : int, optional
        length of baseline period in seconds
    SCS : list, int, optional
        list of (1) Tone and (2) Noise lengths in seconds
    US : int, optional
        length in seconds of US
    ISI : int, optional
        lenth in seconds of ISI
    Trials : int, optional
        numbers of trials
    fs : int, optional
        sampling frequency (default=5 Hz)
    Behav : str
        desired behavioral data ('Freezing' or 'Threshold'; default='Freezing')
    Group: str
        group metadata to assign. mainly useful for within-subjects data where the same subjects
        have different experimental conditions. Leave as default if not within-subjects. 

    Returns
    -------
    BL_df
        a pandas dataframe with the averaged baseline data
    SCS_df
        a pandas dataframe with the averaged SCS data (Tone and Noise)
    Post_df
        a pandas dataframe with both averaged US and ISI data
    """
        
    # Baseline 
    ID = df.xs(Behav).columns                                                   # get IDs
    
    BL_timestamps = [0,BL*fs]                                                   # BL timestamps
    BL_data = df.xs(Behav)[BL_timestamps[0]:BL_timestamps[1]].mean().values     # slice and average data
    
    dict4pandas = {'ID': ID, 'BL': BL_data}                                     # BL dataframe
    BL_df   = pd.DataFrame(dict4pandas)                                     

    
    # Trial prep
    ID_metadata    = np.tile(ID,Trials)                                         # ID metadata
    Trial_metadata = [ele for ele in range(1,Trials+1) for i in range(len(ID))] # trial metadata length of n_rats
    ITI            = SCS[0]+SCS[1]+US+ISI                                      # ITI length

    
    # SCS - Tone 
    CS_metadata   = [ele for ele in ['Tone'] for i in range(len(ID)*Trials)]    # tone metadata length of n_rats*n_trials
    CS_data = slicedata(df, n_trials=Trials, start_time=BL,                     # slice data
                        length = SCS[0], ITI = ITI) 
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata,                  # tone dataframe
                   'CS type': CS_metadata, 'CS Freezing': CS_data}
    Tone_df   = pd.DataFrame(dict4pandas)


    # SCS - Noise
    CS_metadata   = [ele for ele in ['Noise'] for i in range(len(ID)*Trials)]   # noise metadata length of n_rats*n_trials

    start_time = BL + SCS[0]                                                # start time
    CS_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length = SCS[1], ITI = ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata,                  # noise dataframe
                   'CS type': CS_metadata, 'CS Freezing': CS_data}
    Noise_df   = pd.DataFrame(dict4pandas)

    
    # SCS - Total
    SCS_df = Tone_df.append(Noise_df,ignore_index=True)                         # SCS dataframe (tone + noise dfs)


    #  US
    start_time = BL + SCS[0] + SCS[1]                                        # start time
    US_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length = US, ITI = ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'US': US_data}   # US dataframe
    US_df   = pd.DataFrame(dict4pandas)


    #  ISI
    start_time = BL + SCS[0] + SCS[1] + US                                    # start time
    ISI_data = slicedata(df, n_trials=Trials, start_time=start_time,            # slice data
                         length = ISI, ITI = ITI)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'ISI': ISI_data} # ISI dataframe
    ISI_df   = pd.DataFrame(dict4pandas)
    
    Post_df = pd.merge(ISI_df, US_df, on=['ID', 'Trial'], copy='True')          # Post dataframe (ISI + US)
    
    
    #  Group metadata
    if any(Group):
        
        Group_metadata   = [ele for ele in [Group] for i in range(len(ID))]     # group metadata
        
        dict4pandas = {'ID': ID, 'Group': Group_metadata}                       # group dataframe
        Group_df   = pd.DataFrame(dict4pandas)
        
        # merge group df to others
        BL_df   = pd.merge(Group_df, BL_df, on='ID', copy='True')                  # BL + group
        SCS_df  = pd.merge(Group_df, SCS_df, on='ID', copy='True')                 # SCS + group
        Post_df = pd.merge(Group_df, Post_df, on='ID', copy='True')                # Post + group
        
    return BL_df, SCS_df, Post_df
