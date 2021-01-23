import os
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy.signal import medfilt
from tsfresh.feature_extraction import feature_calculators as fc


def extract_feats(ts):
    std = fc.standard_deviation(ts)
    kurtosis = fc.kurtosis(ts)
    skewness = fc.skewness(ts)
    cam = fc.count_above_mean(ts)
    cbm = fc.count_below_mean(ts)
    lsam = fc.longest_strike_above_mean(ts)
    lsbm = fc.longest_strike_below_mean(ts)
    psd = fc.fourier_entropy(ts, bins=1000000)
    energy = fc.abs_energy(ts)
    return np.array([std, kurtosis, skewness, cam, cbm, lsam, lsbm, psd, energy])


def main():
    labels = pd.read_csv(os.getcwd()+'/data/apachePatientResult.csv')
    labels = labels[['patientunitstayid', 'actualicumortality']].replace(['ALIVE', 'EXPIRED'], [0, 1]).drop_duplicates()
    data = dd.read_csv(os.getcwd()+'/data/vitalPeriodic.csv')
    labels = labels.loc[labels['patientunitstayid'].isin(data['patientunitstayid'].unique().compute())]
    ids = labels['patientunitstayid'].to_numpy()
    outcomes = labels['actualicumortality'].to_numpy()
    x = []
    y = []
    block_size = 5000   # Time on a .compute() call is about the same no matter the size of the block, we're just limited by memory
    for block in range(len(labels) // block_size):
        block_time = time.time()
        df = data.loc[data['patientunitstayid'].isin(ids[block*block_size:(block+1)*block_size])].compute()
        for id in range(block_size):
            inner_block = time.time()
            outcome = outcomes[block*block_size:(block+1)*block_size][id]
            ts = df[df['patientunitstayid'] == (ids[block*block_size:(block+1)*block_size][id])]
            ts = ts.sort_values(by='observationoffset')
            t = ts['observationoffset'].to_numpy()
            t = np.cumsum(np.diff(t))
            t = np.hstack((0, t))  # Append zero for first measurement
            hr = ts['heartrate'].to_numpy()
            hr = medfilt(hr, 5)
            ts = pd.DataFrame({'time': t, 'hr': hr})
            ts = ts.interpolate()
            x_i = extract_feats(ts['hr'].to_numpy())
            x.append(x_i)
            y.append(outcome)
            # print('inner block time: {:.2f}'.format(time.time()-inner_block))
        print('done {}/{}'.format((block+1)*block_size, len(labels)))
        print('outer block time: {:.2f}'.format(time.time()-block_time))
        np.save('x.npy', np.vstack(x))
        np.save('y.npy', np.hstack(y))


if __name__ == '__main__':
    main()
