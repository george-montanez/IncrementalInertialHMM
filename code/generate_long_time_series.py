from __future__ import division
import numpy as np
import pylab
from sklearn import preprocessing
from collections import defaultdict
from matplotlib import cm
from InertialHMM import InertialHMM

segment_func = InertialHMM(1,np.random.random(size=(2,2))).get_segments

OUTPUT_DIR = "../data/LongAccelerometerSeries/"

class EndlessData(object):
    def __init__(self):    
        self.datasets = [np.loadtxt("../data/accelerometer_activities/a19/p8/combined.txt", delimiter=",").T,
                        np.loadtxt("../data/accelerometer_activities/a17/p8/combined.txt", delimiter=",").T,
                        np.loadtxt("../data/accelerometer_activities/a18/p8/combined.txt", delimiter=",").T,
                        np.loadtxt("../data/accelerometer_activities/a05/p8/combined.txt", delimiter=",").T,
                        np.loadtxt("../data/accelerometer_activities/a09/p8/combined.txt", delimiter=",").T]
        self.activity_offsets = defaultdict(int)
    
    def get(self, state, num_ticks):
        scaler = preprocessing.StandardScaler()
        scale_func = scaler.fit_transform
        start = self.activity_offsets[state]
        end = self.activity_offsets[state] + num_ticks
        ds = self.datasets[state]
        if end >= ds.shape[1]:
            self.activity_offsets[state] = 0
            start = 0
            end = num_ticks
        self.activity_offsets[state] += num_ticks % ds.shape[1]
        return scale_func(ds[:,start:end])

def generate_accelerometer_datasets(transition_matrix, pi_vector, endless_data_obj, N=10, T=100000):
    ''' Generate data according to defined HMM '''
    A_mat = np.array(transition_matrix) 
    assert(A_mat.shape[0] == len(pi_vector))
    draw_state = lambda p_vec : np.argmax(np.random.multinomial(1, p_vec, size=1))
    i = 0
    while i < N:
        print i
        output_data = []
        output_labels = []
        state = draw_state(pi_vector)
        used_states = set([state,])
        for t in range(T):
            output_labels.append(state)
            output_data.append(endless_data_obj.get(state,1))
            state = draw_state(A_mat[state,:])
            used_states.add(state)
        num_segments = len(segment_func(output_labels))
        print "num_segments", num_segments, "num_states", len(used_states)
        if len(used_states) > 1 and num_segments >= 2:
            np.savetxt(OUTPUT_DIR + './%d_accelerometer_data.txt' % i, np.array(output_data))
            np.savetxt(OUTPUT_DIR + './%d_accelerometer_labels.txt' % i, output_labels, fmt="%d")
            i += 1

if __name__ == '__main__':    
    st = 0.95
    trans_matrix = np.ones((5,5)) * (1. - st) / 4
    np.fill_diagonal(trans_matrix, st)
    pi_vector = np.ones(5) / 5.
    ed = EndlessData()
    generate_accelerometer_datasets(trans_matrix, pi_vector, ed, 10, T=110000)

