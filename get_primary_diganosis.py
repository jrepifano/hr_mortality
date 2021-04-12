# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:42:30 2020

@author: Jake

Process diagnostic information for each known patient and separate into bins
"""
import os
import pandas as pd
import numpy as np
import dask.dataframe as dd


def main():
    labels = pd.read_csv(os.getcwd() + '/data/apachePatientResult.csv')
    labels = labels[['patientunitstayid', 'actualicumortality']].replace(['ALIVE', 'EXPIRED'], [0, 1]).drop_duplicates()
    data = dd.read_csv(os.getcwd()+'/data/vitalPeriodic.csv')
    eicu = labels.loc[labels['patientunitstayid'].isin(data['patientunitstayid'].unique().compute())]
    # eicu = labels['patientunitstayid'].to_numpy()
    diag = pd.read_csv(os.getcwd()+'/data/diagnosis.csv')

    eicu_filt = eicu.loc[eicu['patientunitstayid'].isin(diag['patientunitstayid'])]
    filt_pid = pd.unique(eicu_filt['patientunitstayid'])

    diag_filt = diag.loc[diag['patientunitstayid'].isin(eicu_filt['patientunitstayid'])]
    diag_pid = pd.unique(diag_filt['patientunitstayid'])

    assert(len(filt_pid) == len(diag_pid))
    print('Number of Lost Patients: '+str(len(eicu)-len(eicu_filt)))


    diag_types = ['trauma', 'transplant', 'toxicology', 'surgery', 'renal', 'pulmonary', 'oncology', 'neurlogic',
                  'infectious_diseases', 'hematology', 'genitourinary', 'general', 'gastrointestinal', 'endocrine',
                  'cardiovascular']

    diag_pids = [set() for _ in range(len(diag_types))]
    for i in range(len(diag_filt)):
        idx = np.where(diag_filt.iloc[i, :]['diagnosisstring'].split('|')[0] == diag_types)
        diag_pids[idx[0].item()].add(diag_filt.iloc[i, :]['patientunitstayid'])


if __name__ == '__main__':
    main()
