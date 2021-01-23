import os
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
    x = []
    y = []
    print_iter = 50
    for i in range(len(labels)):
        id = labels['patientunitstayid'].to_numpy()[i]
        outcome = labels['actualicumortality'].to_numpy()[i]
        df = data[data['patientunitstayid'] == id].compute()
        time = df['observationoffset'].to_numpy()
        time = np.cumsum(np.diff(time))
        time = np.hstack((0, time))  # Append zero for first measurement
        hr = df['heartrate'].to_numpy()
        hr = medfilt(hr, 5)
        ts = pd.DataFrame({'time': time, 'hr': hr})
        ts = ts.interpolate()
        x_i = extract_feats(ts['hr'].to_numpy())
        x.append(x_i)
        y.append(outcome)
        # if i % print_iter == 0:
        print('done {}/{}'.format(i, len(labels)))
    np.save('x.npy', np.vstack(x))
    np.save('y.npy', np.hstack(y))


if __name__ == '__main__':
    main()
