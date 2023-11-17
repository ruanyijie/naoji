import h5py, pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d
import os, urllib
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge as RR
from sklearn.metrics import r2_score
from numpy.linalg import inv as inv
plt.switch_backend('agg')

import sys
sys.path.append('/home/renyi/gwx/hmk/DynamicalComponentsAnalysis-main')
def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def load_sabes_data(filename, bin_width_s=.05, high_pass=True, sqrt=True, thresh=5000, zscore_pos=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            print(max_t - t[0])
            n_bins = int(np.floor((max_t - t[0]) / bin_width_s))
            binned_spikes = np.zeros((n_bins, d), dtype=np.int32)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int32)
                    unique_idxs, counts = np.unique(bin_idx, return_counts=True)
                    # make sure to ignore the hash here...
                    binned_spikes[unique_idxs, chan_idx * n_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > thresh]
            result[region] = binned_spikes
        # Get cursor position
        # cursor_pos = f["cursor_pos"][:].T
        # print(len(binned_spikes))
        # # Line up the binned spikes with the cursor data
        # t_mid_bin = np.arange(len(binned_spikes)) * bin_width_s + bin_width_s / 2
        # cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        # cursor_interp = cursor_pos_interp(t_mid_bin)
        # if zscore_pos:
        #     cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
        #     cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        # result["cursor"] = cursor_interp
        return result

def load_data(filename, high_pass=True, sqrt=True, thresh=5000, zscore_pos=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            print(region)
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            min_t = t[0]
            binned_spikes = np.zeros((len(t), d), dtype=np.float32)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times <= max_t]
                    spike_times = spike_times[spike_times >= min_t]
                    # make sure to ignore the hash here...
                    binned_spikes[:len(spike_times), chan_idx * n_sorted_units + unit_idx - 1] = spike_times
            binned_spikes = binned_spikes[:, np.count_nonzero(binned_spikes,axis=0) > thresh]
            
            result[region] = binned_spikes
        
        # Find when target pos changes
        target_pos = f["target_pos"][:].T
        import pandas as pd
        target_pos = pd.DataFrame(target_pos)
        has_change = target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
        time = pd.DataFrame(f['t'][0, :].T)
        # Add start and end times to trial info
        change_times = time.index[has_change]
        start_times = change_times[:-1]
        end_times = change_times[1:]
        # Get target position per trial
        temp_target_pos = target_pos.loc[start_times].to_numpy().tolist()
        # Compute reach distance and angle
        reach_dist = target_pos.loc[end_times - 1].to_numpy() - target_pos.loc[start_times - 1].to_numpy()
        reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
        
        # Create trial info
        result['time'] = time.to_numpy()
        # print(result['time'])
        result['start_times'] = start_times.to_numpy()
        result['end_times'] = end_times.to_numpy()
        result['target_pos'] = temp_target_pos
        result['reach_dist_x'] = reach_dist[:, 0]
        result['reach_dist_y'] = reach_dist[:, 1]
        result['reach_angle'] = reach_angle
        spike_time = []
        for indices in (M1_indices, S1_indices):
            for chan_idx in indices:
                    for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                        spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                        # if spike_times.shape == (2,):
                        #     # ignore this case (no data)
                        #     continue
                        # spike_times = spike_times[0, :]
                        spike_time.append(spike_times)
        
        result['spike_times'] = np.array(spike_time,dtype=object)
        # print(result['spike_times'])
        result['vels'] = []
        cursor_pos = f['cursor_pos'][:].T
        vel = np.diff(cursor_pos,axis=0,prepend=cursor_pos[0].reshape(1,-1))/0.004
        vel[0]=vel[1]
        # print(vel)
        acc = vel = np.diff(vel,axis=0,prepend=vel[0].reshape(1,-1))/0.004
        acc[0]=acc[2]
        acc[1]=acc[2]
        result['vels'] = vel
        result['accs'] = acc
        print(len(vel))
        print(len(result['time']))
        print(len(result['reach_angle']))
        for i in range (0,len(result['reach_angle'])):
            vel = []
            if(i==0):
                vel.append((result['reach_dist_x'][1]-result['reach_dist_x'][0])/0.004)
                vel.append((result['reach_dist_y'][1]-result['reach_dist_y'][0])/0.004)
            else:
                vel.append((result['reach_dist_x'][i]-result['reach_dist_x'][i-1])/0.004)
                vel.append((result['reach_dist_x'][i]-result['reach_dist_x'][i-1])/0.004)
            result['vels'].append(vel)
            if(i==0):
                result['vels'].append(math.sqrt((result['reach_dist_x'][1]-result['reach_dist_x'][0])*(result['reach_dist_x'][1]-result['reach_dist_x'][0])+(result['reach_dist_y'][1]-result['reach_dist_y'][0])*(result['reach_dist_y'][1]-result['reach_dist_y'][0])/0.004))
                continue
            result['vels'].append(math.sqrt((result['reach_dist_x'][i]-result['reach_dist_x'][i-1])*(result['reach_dist_x'][i]-result['reach_dist_x'][i-1])+(result['reach_dist_y'][i]-result['reach_dist_y'][i-1])*(result['reach_dist_y'][i]-result['reach_dist_y'][i-1])/0.004))
        print(result['vels'])                           
        return result

def plot_tuning_curve(data, cell=0):
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
    pi = np.pi
    n_bins = int(12)
    binned_spikes = np.zeros(n_bins, dtype=np.int32)
    x = []
    for i in range(12):
        if i==6:
            start_angle1 = -15. + 6 * 30.
            end_angle1 = 15. - 6 * 30.
            index = np.where(((start_angle1 < reach_angle) & (180. >= reach_angle)) | ((-180. < reach_angle) & (end_angle1 >= reach_angle)))
            x.append(i*30)
        elif i < 6:
            start_angle = -15. + i * 30.
            end_angle = 15. + i * 30.
            index = np.where((start_angle < reach_angle) & (end_angle >= reach_angle))
            x.append(i*30)
        else:
            start_angle = -360. + i * 30. - 15.
            end_angle = -360. + i * 30. + 15.
            index = np.where((start_angle < reach_angle) & (end_angle >= reach_angle))
            x.append(i*30 - 360)

        start_time = start_times[index]
        end_time = end_times[index]

        raster = []
        for temp_i in range(len(index[0])):
            temp_raster = X[cell]
            start_timestamp = start_time[temp_i] * 0.004 + t_min 
            end = end_time[temp_i] if (end_time[temp_i] - start_time[temp_i]) < 1000 else (1000 + start_time[temp_i])
            end_timestamp = end * 0.004 + t_min
            temp_raster = temp_raster[start_timestamp < temp_raster]
            temp_raster = temp_raster[temp_raster < end_timestamp]
            binned_spikes[i] += temp_raster.shape[0]
    x = np.array(x)
    index = np.where(binned_spikes>0)
    y = binned_spikes[index]
    x = x[index]
    def target_func(x, a0, a2, a3):
        return a0 * np.sin((x + a2)/360*2*pi) + a3
    # a0*sin(a1*x+a2)+a3
    import scipy.optimize as optimize
    a0 = (max(y) - min(y)) / 2
    max_index = y.tolist().index(max(y))
    a2 = 90 - x[max_index]
    a3 = a0
    p0 = [a0, a2, a3]
    para, _ = optimize.curve_fit(target_func, x, y, p0=p0)
    print(para)
    y_fit = [target_func(a, *para) for a in x]
    #Get metric of fit
    y_mean=np.mean(y)
    R2=1-np.sum((y_fit-y)**2)/np.sum((y-y_mean)**2)
    print('R2s:', R2)
    # plt.figure() #初始化一张图
    # plt.scatter(y, x, c='red', label='function')  # 标签 即为点代表的意思
    # # 3.展示图形
    # plt.legend()  # 显示图例
    # plt.xlabel('angle')
    # plt.ylabel('trail') 
    # plt.title(f'{cell}_tuning_curve') 
    # plt.savefig(f"./{cell}_tuning_curve.jpg")
    return R2


def plot_raster(data, direction):
    dir2an = {'right': 0.,
              'left': 180.,
              'up': 90.,
              'down': -90.,
    }
    angle = dir2an[direction]
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
        
    index = np.where(reach_angle == angle)
    start_times = start_times[index]
    end_times = end_times[index]
    raster = []
    for i in range(len(index[0])):
        temp_raster = X[0]
        start_timestamp = start_times[i] * 0.004 + t_min 
        end_timestamp = end_times[i] * 0.004 + t_min
        temp_raster = temp_raster[start_timestamp - 20 *0.004 < temp_raster]
        temp_raster = temp_raster[temp_raster < end_timestamp]
        trans_raster = temp_raster - start_timestamp
        if trans_raster.shape[0] > 0:
            raster.append(trans_raster)

    plt.eventplot(raster[:][:100])
    plt.xlabel('Time (s)')
    plt.ylabel('trail')
    plt.savefig(f"./{direction}_raster.jpg")
    plt.show()

def plot_psth(data, direction, bin_width_s=0.04):
    dir2an = {'right': 0.,
              'left': 180.,
              'up': 90.,
              'down': -90.,
    }
    angle = dir2an[direction]
    X = data['M1'].T 
    reach_angle = data['reach_angle']
    start_times = data['start_times']
    end_times = data['end_times']
    time = data['time']
    t_min = time[0]
        
    index = np.where(reach_angle == angle)
    start_times = start_times[index]
    end_times = end_times[index]
    raster = []
    max = 0
    min = 2
    for i in range(len(index[0])):
        temp_raster = X[0]
        start_timestamp = start_times[i] * 0.004 + t_min 
        end_timestamp = end_times[i] * 0.004 + t_min
        temp_raster = temp_raster[start_timestamp - 20 *0.004 < temp_raster]
        temp_raster = temp_raster[temp_raster < end_timestamp]
        trans_raster = temp_raster - start_timestamp
        if trans_raster.shape[0] > 0:
            max = max if trans_raster[0].max() < max else trans_raster[0].max()
            min = min if trans_raster[0].min() > max else trans_raster[0].min()
            raster += trans_raster.tolist()

    n_bins = int(np.floor((max - min) / bin_width_s))
    # binned_spikes = np.zeros(n_bins, dtype=np.int32)
    # for spike_times in raster:
    #     spike_times = spike_times[spike_times - min < n_bins * bin_width_s]
    #     bin_idx = np.floor((spike_times - min) / bin_width_s).astype(np.int32)
    #     unique_idxs, counts = np.unique(bin_idx, return_counts=True)
    #     binned_spikes[unique_idxs] += counts
    
    plt.figure() #初始化一张图
    plt.hist(raster, n_bins)  #直方图关键操作
    plt.xlabel('number')
    plt.ylabel('trail') 
    plt.title(f'{direction}_psth') 
    plt.savefig(f"./{direction}_psth.jpg")

def bin_spikes(spike_times,dt,wdw_start,wdw_end):
    """
    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=len(spike_times) #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data

def bin_output(outputs,output_times,dt,wdw_start,wdw_end,downsample_factor=1):
    """
    Function that puts outputs into bins

    Parameters
    ----------
    outputs: matrix of size "number of times the output was recorded" x "number of features in the output"
        each entry in the matrix is the value of the output feature
    output_times: a vector of size "number of times the output was recorded"
        each entry has the time the output was recorded
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for binning the outputs
    wdw_end: number (any format)
        the end time for binning the outputs
    downsample_factor: integer, optional, default=1
        how much to downsample the outputs prior to binning
        larger values will increase speed, but decrease precision

    Returns
    -------
    outputs_binned: matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    ###Downsample output###
    #We just take 1 out of every "downsample_factor" values#
    if downsample_factor!=1: #Don't downsample if downsample_factor=1
        downsample_idxs=np.arange(0,output_times.shape[0],downsample_factor) #Get the idxs of values we are going to include after downsampling
        outputs=outputs[downsample_idxs,:] #Get the downsampled outputs
        output_times=output_times[downsample_idxs] #Get the downsampled output times

    ###Put outputs into bins###
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    output_dim=outputs.shape[1] #Number of output features
    outputs_binned=np.empty([num_bins,output_dim]) #Initialize matrix of binned outputs
    outputs_array = np.array(outputs)
    output_times_array = np.array(output_times)
    #Loop through bins, and get the mean outputs in those bins
    for i in range(num_bins): #Loop through bins
        idxs=np.where((np.squeeze(output_times_array)>=edges[i]) & (np.squeeze(output_times_array)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
        for j in range(output_dim): #Loop through output features
            outputs_binned[i,j]=np.mean(outputs[idxs,j])
        # outputs_binned[i, :] = np.mean(outputs_array[idxs, :], axis=0)

    return outputs_binned

class KalmanFilterRegression(object):

    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):

        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)

        #number of time bins
        nt=X.shape[1]

        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        #In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) #Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt #Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        A,W,H,Q=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))

        #Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            #Do first part of state update - based on transition matrix
            P_m=A*P*A.T+W
            state_m=A*state

            #Do second part of state update - based on measurement matrix
            K=P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            P=(np.matrix(np.eye(num_states))-K*H)*P_m
            state=state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1]=np.squeeze(state) #Record state at the timestep
        y_test_predicted=states.T
        return y_test_predicted

KalmanFilterDecoder = KalmanFilterRegression

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s




########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos

def kf(filename,bin_width_s=.05, thresh=5000):
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            min_t = t[0]
            binned_spikes = []
            num = 0
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times <= max_t]
                    spike_times = spike_times[spike_times >= min_t]
                    # make sure to ignore the hash here...
                    if len(spike_times) > thresh:
                        binned_spikes.append(spike_times)
        t_start=min_t
        t_end=max_t
        downsample_factor=1
        #Bin neural data using "bin_spikes" function
        neural_data=bin_spikes(binned_spikes, bin_width_s, t_start, t_end)
        print(neural_data.shape)
        # get vel 
        cursor_pos = f["cursor_pos"][:].T
        vel = np.diff(cursor_pos, axis=0, prepend=cursor_pos[0].reshape(1,-1)) / 0.004
        vel[0] = vel[1]
        acc = np.diff(vel, axis=0, prepend=vel[0].reshape(1,-1)) / 0.004
        acc[0] = acc[2]
        acc[1] = acc[2]

        output = np.concatenate((cursor_pos,vel, acc), axis=1)
        vels_binned=bin_output(output,t,bin_width_s,t_start,t_end,downsample_factor)
        print(vels_binned.shape)

    #Bin output (velocity) data using "bin_output" function
    import pickle

    # data_folder='E:\data/'

    # with open(data_folder+'example_data_s1.pickle','wb') as f:
    #     pickle.dump([neural_data,vels_binned],f)
    lag=0 #What time bin of spikes should be used relative to the output
    #(lag=-1 means use the spikes 1 bin before the output)
    #The covariate is simply the matrix of firing rates for all neurons over time
    X_kf=neural_data

    # For the Kalman filter, we use the position, velocity, and acceleration as outputs
    # Ultimately, we are only concerned with the goodness of fit of velocity (for this dataset)
    # But using them all as covariates helps performance

    # We will now determine position
    # pos_binned=np.zeros(vels_binned.shape) #Initialize 
    # pos_binned[0,:]=0 #Assume starting position is at [0,0]
    # #Loop through time bins and determine positions based on the velocities
    # for i in range(pos_binned.shape[0]-1): 
    #     pos_binned[i+1,0]=pos_binned[i,0]+vels_binned[i,0]*.05 #Note that .05 is the length of the time bin
    #     pos_binned[i+1,1]=pos_binned[i,1]+vels_binned[i,1]*.05

    # #We will now determine acceleration    
    # temp=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins 
    # acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

    # #The final output covariates include position, velocity, and acceleration
    # y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=0)
    y_kf = vels_binned[:, 0:6]
    num_examples=X_kf.shape[0]

    #Re-align data to take lag into account
    if lag<0:
        y_kf=y_kf[-lag:,:]
        X_kf=X_kf[0:num_examples+lag,:]
    if lag>0:
        y_kf=y_kf[0:num_examples-lag,:]
        X_kf=X_kf[lag:num_examples,:]

    #Set what part of data should be part of the training/testing/validation sets
    training_range=[0, 0.7]
    testing_range=[0.7, 0.85]
    valid_range=[0.85,1]

    #Number of examples after taking into account bins removed for lag alignment
    num_examples_kf=X_kf.shape[0]
            
    #Note that each range has a buffer of 1 bin at the beginning and end
    #This makes it so that the different sets don't include overlapping data
    training_set=np.arange(np.int32(np.round(training_range[0]*num_examples_kf))+1,np.int32(np.round(training_range[1]*num_examples_kf))-1)
    testing_set=np.arange(np.int32(np.round(testing_range[0]*num_examples_kf))+1,np.int32(np.round(testing_range[1]*num_examples_kf))-1)
    valid_set=np.arange(np.int32(np.round(valid_range[0]*num_examples_kf))+1,np.int32(np.round(valid_range[1]*num_examples_kf))-1)

    #Get training data
    X_kf_train=X_kf[training_set,:]
    y_kf_train=y_kf[training_set,:]

    #Get testing data
    X_kf_test=X_kf[testing_set,:]
    y_kf_test=y_kf[testing_set,:]

    #Get validation data
    X_kf_valid=X_kf[valid_set,:]
    y_kf_valid=y_kf[valid_set,:]

    #Z-score inputs 
    X_kf_train_mean=np.nanmean(X_kf_train,axis=0)
    X_kf_train_std=np.nanstd(X_kf_train,axis=0)
    X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std
    X_kf_test=(X_kf_test-X_kf_train_mean)/X_kf_train_std
    X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std

    #Zero-center outputs
    y_kf_train_mean=np.mean(y_kf_train,axis=0)
    y_kf_train=y_kf_train-y_kf_train_mean
    y_kf_test=y_kf_test-y_kf_train_mean
    y_kf_valid=y_kf_valid-y_kf_train_mean

    #Declare model
    model_kf=KalmanFilterDecoder(C=1) #There is one optional parameter that is set to the default in this example (see ReadMe)

    #Fit model
    model_kf.fit(X_kf_train,y_kf_train)

    #Get predictions
    y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)

    #Get metrics of fit (see read me for more details on the differences between metrics)
    #First I'll get the R^2
    print(y_kf_valid)
    print(len(y_kf_valid))
    print(y_valid_predicted_kf)
    print(len(y_valid_predicted_kf))
    R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
    print('R2:',R2_kf[0:6]) #I'm just printing the R^2's of the 3rd and 4th entries that correspond to the velocities
    #Next I'll get the rho^2 (the pearson correlation squared)
    rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
    print('rho2:',rho_kf[0:6]**2) #I'm just printing the rho^2's of the 3rd and 4th entries that correspond to the velocities

    #As an example, I plot an example 1000 values of the x velocity (column index 2), both true and predicted with the Kalman filter
    #Note that I add back in the mean value, so that both true and predicted values are in the original coordinates
    fig_x_kf=plt.figure()
    # plt.plot(y_kf_valid[1000:2000,2]+y_kf_train_mean[2],'b')
    # plt.plot(y_valid_predicted_kf[1000:2000,2]+y_kf_train_mean[2],'r')
    # plt.savefig(f"./kf.jpg")
    # plt.show()
    #Save figure
    # fig_x_kf.savefig('x_velocity_decoding.eps')

def pca_and_sort(filename, chan_idx):
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)

        # Perform binning
        n_sorted_units = f["wf"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
        unsorted_spike = f[f["wf"][0, 5]][()].T
            
        
    
        from sklearn.decomposition import PCA
        import pandas as pd
        import numpy as np
        # x_values = np.arange(0, 1.92, 0.04)
        # for num in range(128):
        #     y_values = unsorted_spike[num]
        #     # 将点绘制在图上
        #     # plt.scatter(x_values, y_values)	
        #     # 将点连起来
        #     plt.plot(x_values, y_values)
        # plt.savefig(f"./{chan_idx}_spike.jpg")

        pca_sk = PCA(n_components=8)
        newMat = pca_sk.fit_transform(unsorted_spike)
        # plt.scatter(newMat[:, 0], newMat[:, 1],marker='o')
        # plt.savefig(f"./{chan_idx}_pca.jpg")

        from sklearn.cluster import KMeans
        n_clusters = 4
        cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(newMat)
        index = cluster.predict(newMat)
        # plt.scatter(newMat[:, 0], newMat[:, 1], c=index)
        # plt.savefig(f"./{chan_idx}_kmeans.jpg")

        gini_index = 0
        dataset_len = 0
        len = []
        gini_len = []
        for unit_idx in range(1, n_sorted_units + 1):  # ignore hash!
            spike_times = f[f["wf"][unit_idx, chan_idx]][()].T
            if spike_times.shape[0] < 2000:
                continue
            if spike_times.shape == (2,):
                # ignore this case (no data)
                continue
            dataset_len += spike_times.shape[0]
            len.append(spike_times.shape[0])
            temp_mat = pca_sk.transform(spike_times)
            temp = cluster.predict(temp_mat)
            nique_idxs, counts = np.unique(temp, return_counts=True)
            print(nique_idxs, counts)
            temp_p = counts / np.sum(counts)
            gini = 1 - np.sum(temp_p**2)
            gini_len.append(gini)
        pre = np.array(len) / dataset_len
        gini_index = pre * (np.array(gini_len).reshape(1,-1))
        print(np.sum(gini_index))
                
## Make trial data
if __name__ == '__main__':
    fname = 'D:\hw2dataset/indy_20160407_02.mat'
    data = load_data(fname)
    # print(data)
    # # 画神经元raster图
    # plot_raster(data, 'down')
    # # 画神经元PSTH图
    # plot_psth(data, 'down')
    # 画神经元tuning curve图
    # R2s = []
    # for num in range(100):
    #     R2 = plot_tuning_curve(data, num)
    #     unsort = int(R2/0.05)
    #     R2s.append(R2)
    # plt.figure() #初始化一张图
    # plt.hist(R2s, 20, range=(0,1))  #直方图关键操作
    # plt.xlabel('R2')
    # plt.ylabel('cell') 
    # plt.savefig(f"./check_R2.jpg")
    # kf(fname)
    # 神经元PCA降维和分类
    # pca_and_sort(fname, 22)



    