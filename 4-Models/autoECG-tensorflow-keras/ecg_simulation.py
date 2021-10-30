#!/usr/bin/env python
# coding: utf-8

# In[5]:


import neurokit2 as nk
import numpy as np
import pandas as pd
import parameters as para


# In[26]:


def simulation(normal_N,abnormal_N, save_params = False):
    normal_data = []
    normal_params = []
    print('Creating normal dataset')
    for i in range(normal_N):
        ti = np.random.normal(para.mu_t_1, para.sigma_t_1)
        ai = np.random.normal(para.mu_a_1, para.sigma_a_1)
        bi = np.random.normal(para.mu_b_1, para.sigma_b_1)
        hr = np.random.normal(para.mu_hr_1, para.sigma_hr_1)
        noise = np.random.uniform(low=para.min_noise_1, high=para.max_noise_1)
        ecgs, _ = nk.ecg_simulate(duration=para.duration*2, sampling_rate=para.sampling_rate, noise=noise, Anoise=para.Anoise, heart_rate=hr, gamma=para.gamma, ti=ti, ai=ai, bi=bi)
        ecgs = np.array(ecgs)
        start_i = np.random.randint(len(ecgs[0])//4, len(ecgs[0])//2)
        normal_data.append(ecgs[:,start_i:start_i+para.sampling_rate*para.duration])
        normal_params.append({'ti':ti, 'ai':ai, 'bi':bi, 'hr':hr, 'noise':noise, 'gamma': para.gamma})

    abnormal_data = []
    abnormal_params = []
    print('Creating abnormal dataset')
    for i in range(abnormal_N):
        ti = np.random.normal(para.mu_t_2, para.sigma_t_2)
        ai = np.random.normal(para.mu_a_2, para.sigma_a_2)
        bi = np.random.normal(para.mu_b_2, para.sigma_b_2)
        hr = np.random.normal(para.mu_hr_2, para.sigma_hr_2)
        noise = np.random.uniform(low=para.min_noise_2, high=para.max_noise_2)
        ecgs, _ = nk.ecg_simulate(duration=para.duration*2, sampling_rate=para.sampling_rate, noise=noise, Anoise=para.Anoise, heart_rate=hr, gamma=para.gamma, ti=ti, ai=ai, bi=bi)
        ecgs = np.array(ecgs)
        start_i = np.random.randint(len(ecgs[0])//4, len(ecgs[0])//2)
        abnormal_data.append(ecgs[:,start_i:start_i+para.sampling_rate*para.duration])
        abnormal_params.append({'ti':ti, 'ai':ai, 'bi':bi, 'hr':hr, 'noise':noise, 'gamma': para.gamma})

    labels = np.array([0]*len(normal_data) + [1]*len(abnormal_data))
    permutation = np.random.permutation(len(labels))
    data = np.array(normal_data+abnormal_data)
    data_params = np.array(normal_params+abnormal_params)
    labels = labels[permutation]
    data = data[permutation]
    data_params = data_params[permutation]

    np.save('sim_ecg_data', data) # save ECG data
    np.save('sim_ecg_labels', labels) # save label
    if (save_params):
        np.save('sim_ecg_params',data_params) # save parameters for each ecg sample (12,2500)

