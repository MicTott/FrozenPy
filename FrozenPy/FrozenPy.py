# ====================
#  Necessary packages
# ====================

import numpy as np
import pandas as pd
import os

# ============================
#  Functions for loading data
# ============================

def read_out(DIR_NAME, prefix='Rat'):

    """
    Reads all MedPC .out files from the input directory and saves 
    as .raw.csv file in the same directory.

    Note: This file does not resave files if they already exist. To rewrite
    files the existing file must be deleted.

    Parameters
    ----------
    DIR_NAM : str
        directory containing .out files
    prefix : str, optional (default ='Rat')
        desired ID prefix. (ex. 'Rat' for Rat1, Rat2, Rat3, etc.)

    Returns
    -------
    This function does not return variables, but instead saves the read data
    into a new .raw.csv file.
    """
        
    os.chdir(DIR_NAME)
    for file in os.listdir(DIR_NAME):
        if (file.endswith('.out')):

            columns = _read_names(DIR_NAME, file, prefix)
            names_idx = [i for i, s in enumerate(columns) if not 'DELETE' in s]
            names = [columns[i] for i in names_idx]

            data = _read_data(DIR_NAME, file, names_idx)
            df = pd.DataFrame(data=data, columns=names)

            NewCSVname = os.path.splitext(file)[0] + '.raw.csv'
            if not os.path.isfile(NewCSVname):
                df.to_csv(NewCSVname, index=False)
                print('Saved ' + NewCSVname +'!')
            else:
                print('"' + NewCSVname + '" already exists. Delete to rewrite file.')

    print('Finished reading .out files')


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

    rawcsv_df = pd.read_csv(fname, sep=',')

    if delete.lower() in ['y', 'yes']:
        rawcsv_df.drop(rawcsv_df.filter(regex='DELETE'),1,inplace=True)
        print('Dropped "DELETE" columns in ' + fname)
    else:
        print('Did not drop "DELETE" columns in ' + fname)

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

def _read_names(DIR_NAME, file, prefix):
    """
    Reads the column names from .out files and adds a prefix if none exists

    Parameters
    ----------
    DIR_NAM : str
        directory containing .out files
    file : str
        .out file to read
    prefix : str, optional (default ='Rat')
        desired ID prefix. (ex. 'Rat' for Rat1, Rat2, Rat3, etc.)

    Returns
    -------
    names : list
        list of column names
    """
    # read row with names from .out
    names = pd.read_csv(os.path.join(DIR_NAME, file),
                        engine   = 'python',
                        skiprows = 10,
                        nrows = 1,
                        header= 0,
                        sep = 'Subject ID ',
                        skipinitialspace = True)
    names = names.drop(names.columns[0], axis=1) # drop first blank column
    names = names.columns.ravel() # remove brackets from numbers
    
    # if int change to char and add prefix
    if names.dtype.type is np.int_:
            names = np.char.mod('%d', names)
            names = [prefix + i for i in names]
            
    # np array to list and remove whitespace
    names = names.tolist()
    names = [x.strip() for x in names]
    return names

def _read_data(DIR_NAME, file, index):
    
    """
    Reads the data from specified columns in .out files

    Parameters
    ----------
    DIR_NAM : str
        directory containing .out files
    file : str
        .out file to read
    index : list
        indexes of the columns containing behavioral data.
        (i.e. not 'DELETE' columns)

    Returns
    -------
    data : array
        an array (matrix) of threshold data from .out files
    """
    
    df = pd.read_csv(os.path.join(DIR_NAME, file),
                     delimiter='\t',
                     skiprows = 13,
                     header=None,)
    df = df.drop(df.columns[0], axis=1) # drop first column (blank)
    data = df.iloc[:, index].values # only get desired indexes
    
    return data

# ==================================
#  Functions for detecting freezing
# ==================================

def detect_freezing(Threshold_df, threshold = 10):

    """
    Detects freezing using a threshold from raw motion (MedPC Threshold) data. This function
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

    Threshold_values = Threshold_df.values
    rows = np.size(Threshold_values, 0)
    columns = np.size(Threshold_values, 1)

    Freezing_np = np.zeros((rows, columns))
    for c in range(columns):
        for r in range(5, rows):
            data2check = Threshold_values[r-5:r, c] # 1 second of data to check

            if all(n <= threshold  for n in data2check):
                Freezing_np[r, c] = 1

    Freezing_df = pd.DataFrame(Freezing_np)
    Corrected_Freezing = _correct_freezing(Freezing_df)

    Corrected_Freezing.columns = Threshold_df.columns
    Corrected_Freezing = Corrected_Freezing.multiply(100)

    Behav_df = pd.concat([Threshold_df, Corrected_Freezing],keys=['Threshold', 'Freezing'])

    return Behav_df

def _correct_freezing(Freezing_df):

    """
    Corrects freezing detected by detectFreezing(), which only defines 
    freezing one sample at a time,and thus cannot account for freezing onset in
    which 1 second of freezing must be counted. For example, at freezing onset 
    5 samples (1 sec) must be below below threhsold values, but only detectFreezing()
    only defines 0.2 sec of freezing at time. So, this function looks for 
    freezing onset and changes that '1' to a '5' to account for this.

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
    Freezing_values = Freezing_df.values
    rows = np.size(Freezing_values,0)
    columns = np.size(Freezing_values,1)

    # correct freezing
    Freezing_final = np.zeros((rows,columns))
    for c in range(0,columns):

        for r in range(4,rows):
            previous = Freezing_values[r-1,c]
            current = Freezing_values[r,c]

            if current == 0:
                Freezing_final[r, c] = 0
            elif current == 1 and previous == 0:
                Freezing_final[r, c] = 5
            elif current == 1 and previous == 1:
                Freezing_final[r, c] = 1

    Corrected_Freezing_df = pd.DataFrame(Freezing_final)

    return Corrected_Freezing_df

# ============================
#  Functions for slicing data
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
        timestamps[trial] = [start_time+ITI*trial, start_time+length+ITI*trial] # start, stop timestamps

    # slice data with timestamps and average      
    final_data = np.array([])                                         # initialize
    for (start, stop) in timestamps:                                  # loop through timestamps
        averaged_trial = df.xs(Behav)[start*fs+1:stop*fs+1].mean().values # slice and average
        final_data = np.append(final_data, averaged_trial)           # append

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
    BL_df = pd.DataFrame(dict4pandas)                                     


    # Trial prep
    ID_metadata = np.tile(ID,Trials)                                         # ID metadata
    Trial_metadata = [ele for ele in range(1,Trials+1) for i in range(len(ID))] # trial metadata length of n_rats
    ITI = CS+US+ISI                                                  # ITI length

    # CS
    CS_data = slicedata(df, n_trials=Trials, start_time=BL,                     # slice data
                        length=CS, ITI=ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'CS': CS_data}   # CS dataframe
    CS_df = pd.DataFrame(dict4pandas)


    # US
    start_time = BL + CS                                          # start time
    US_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length=US, ITI=ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'US': US_data}   # US dataframe
    US_df = pd.DataFrame(dict4pandas)


    # ISI
    start_time = BL + CS + US                                                   # start time
    ISI_data = slicedata(df, n_trials=Trials, start_time=start_time,            # slice data
                         length = ISI, ITI = ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'ISI': ISI_data} # ISI dataframe
    ISI_df = pd.DataFrame(dict4pandas)

    # Make Trials df
    Trials_df = pd.merge(CS_df, US_df, on=['ID', 'Trial'], copy='True')         # combine CS and US data
    Trials_df = pd.merge(Trials_df, ISI_df, on=['ID', 'Trial'], copy='True')    # add ISI data

    # Add Group metadata, if any
    if any(Group):

        Group_metadata = [ele for ele in [Group] for i in range(len(ID))]     # group metadata

        dict4pandas = {'ID': ID, 'Group': Group_metadata}                       # group dataframe
        Group_df = pd.DataFrame(dict4pandas)

        # merge group df to others
        BL_df = pd.merge(Group_df,BL_df,on='ID',copy='True')               # BL + group
        Trials_df = pd.merge(Group_df,Trials_df,on='ID',copy='True')           # Trials + group

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
    BL_df = pd.DataFrame(dict4pandas)                                     


    # Trial prep
    ID_metadata = np.tile(ID,Trials)                                         # ID metadata
    Trial_metadata = [ele for ele in range(1,Trials+1) for i in range(len(ID))] # trial metadata length of n_rats
    ITI = SCS[0]+SCS[1]+US+ISI                                      # ITI length


    # SCS - Tone 
    CS_metadata = [ele for ele in ['Tone'] for i in range(len(ID)*Trials)]    # tone metadata length of n_rats*n_trials
    CS_data = slicedata(df, n_trials=Trials, start_time=BL,                     # slice data
                        length = SCS[0], ITI = ITI, Behav=Behav) 
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata,                  # tone dataframe
                   'CS type': CS_metadata, 'CS Freezing': CS_data}
    Tone_df = pd.DataFrame(dict4pandas)


    # SCS - Noise
    CS_metadata = [ele for ele in ['Noise'] for i in range(len(ID)*Trials)]   # noise metadata length of n_rats*n_trials

    start_time = BL + SCS[0]                                                # start time
    CS_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length = SCS[1], ITI = ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata,                  # noise dataframe
                   'CS type': CS_metadata, 'CS Freezing': CS_data}
    Noise_df = pd.DataFrame(dict4pandas)


    # SCS - Total
    SCS_df = Tone_df.append(Noise_df,ignore_index=True)                         # SCS dataframe (tone + noise dfs)


    # US
    start_time = BL + SCS[0] + SCS[1]                                        # start time
    US_data = slicedata(df, n_trials=Trials, start_time=start_time,             # slice data
                        length = US, ITI = ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'US': US_data}   # US dataframe
    US_df = pd.DataFrame(dict4pandas)


    # ISI
    start_time = BL + SCS[0] + SCS[1] + US                                    # start time
    ISI_data = slicedata(df, n_trials=Trials, start_time=start_time,            # slice data
                         length = ISI, ITI = ITI, Behav=Behav)
    dict4pandas = {'ID': ID_metadata, 'Trial': Trial_metadata, 'ISI': ISI_data} # ISI dataframe
    ISI_df  = pd.DataFrame(dict4pandas)

    Post_df = pd.merge(ISI_df, US_df, on=['ID', 'Trial'], copy='True')          # Post dataframe (ISI + US)


    # Group metadata
    if any(Group):

        Group_metadata = [ele for ele in [Group] for i in range(len(ID))]     # group metadata

        dict4pandas = {'ID': ID, 'Group': Group_metadata}                       # group dataframe
        Group_df = pd.DataFrame(dict4pandas)

        # merge group df to others
        BL_df = pd.merge(Group_df, BL_df, on='ID', copy='True')                  # BL + group
        SCS_df = pd.merge(Group_df, SCS_df, on='ID', copy='True')                 # SCS + group
        Post_df = pd.merge(Group_df, Post_df, on='ID', copy='True')                # Post + group

    return BL_df, SCS_df, Post_df
