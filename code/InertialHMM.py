from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from gini import GRLC

MIN_START_PROB = 1e-200
MIN_TRANS_PROB = 1e-200
MIN_ALPHA_BETA = 1e-250
MIN_D2_VAL = 1e-200
MIN_COV_VAL = 1e-3
MAX_ITER = 10
MIN_GINI = .5

class InertialHMM(object):
    class RgznModes(object):
        def __init__(self):
            self.STANDARD = 0
            self.MAP = 1
            self.INERTIAL = 2
            self.MAP_SCALE_FREE = 3

    '''****************************'''        
    '''       Initialization       '''
    '''****************************'''

    def __init__(self, number_of_states, sequence, regularization_mode = 0, window_width=30):        
        self.num_states = number_of_states
        emission_means, emission_covariances = self.get_kmeans_emission_init(sequence)
        self.emission_density_objs = [multivariate_normal(mean=emission_means[k], cov=emission_covariances[k]) for k in range(number_of_states)]
        self.prev_emission_means = emission_means
        self.prev_emission_covariances = emission_covariances
        start_probs, transition_probs = self.get_random_start_and_trans_probs()        
        self.start_probs = start_probs
        self.trans_probs = transition_probs
        self.trans_update_method = [self.update_transition_probs_standard,
                                    self.update_transition_probs_MAP,
                                    self.update_transition_probs_inertial_regularized,
                                    self.update_transition_probs_MAP_scale_free,][regularization_mode]
        self.scaling_factors = None
        self.alpha_table = None
        self.beta_table = None
        self.gamma_table = None
        self.xi_table = None    
        self.current_avg_ll = None
        self.log_likelihood = float('-inf')
        self.t = 0
        self.data_buffer = []
        self.previous_state_path_masses = None
        self.window_width = window_width

    def get_random_start_and_trans_probs(self):
        K = self.num_states
        sub_table = np.random.random(size=(K,K))
        sub_table = (sub_table.T / sub_table.sum(axis=1)).T
        return sub_table[0], sub_table

    def get_kmeans_emission_init(self, sequence):
        K = self.num_states
        assignments = KMeans(n_clusters=K).fit_predict(sequence)
        means = []
        covs = []
        for k in range(K):
            points = sequence[np.equal(assignments, k)]
            mean_vector = np.mean(points, axis=0)
            cov_matrix = np.cov(points, rowvar=0)
            cov_matrix[cov_matrix==0] = MIN_COV_VAL
            means.append(mean_vector)
            covs.append(cov_matrix)
        return means, covs

    '''****************************'''        
    '''     Gaussian Emissions     '''
    '''****************************'''

    def get_emission_prob(self, j, point):
        return self.emission_density_objs[j].pdf(point)
    
    def get_log_emission_prob(self, j, point):
        return self.emission_density_objs[j].logpdf(point)

    '''****************************'''        
    '''     Forward / Backward     '''
    '''****************************'''

    def forward(self, sequence):
        ''' Implements the Forward Algorithm, creates alpha table. '''
        T = len(sequence)
        K = self.num_states
        x = sequence
        alpha_table = np.zeros((T, K), dtype="float64")
        scaling_factors = np.zeros(T)
        for k in range(K):
            alpha_table[0][k] = self.start_probs[k] * self.get_emission_prob(k, x[0])
        if not np.all(alpha_table[0]):
            for k in range(K):
                alpha_table[0][k] = self.start_probs[k] * self.get_emission_prob(k, x[0]) + MIN_ALPHA_BETA
        scaling_factors[0] = alpha_table[0].sum() 
        alpha_table[0,:] /= scaling_factors[0]
        for t in range(1, T):
            for k in range(K):
                alpha_table[t][k] = np.dot(alpha_table[t-1,:], self.trans_probs[:,k]) * self.get_emission_prob(k, x[t])
            if not np.all(alpha_table[t]):
                for k in range(K):
                    alpha_table[t][k] = np.dot(alpha_table[t-1,:], self.trans_probs[:,k]) * self.get_emission_prob(k, x[t]) + MIN_ALPHA_BETA
            scaling_factors[t] = alpha_table[t,:].sum()  
            alpha_table[t,:] /= scaling_factors[t]
        ''' save results '''
        self.alpha_table = alpha_table
        self.scaling_factors = scaling_factors       

    def backward(self, sequence):
        ''' Implements the Backward algorithm, creates beta table. '''
        T = len(sequence)
        K = self.num_states
        x = sequence
        beta_table = np.zeros((T, K), dtype="float64")
        beta_table[-1, :] = 1.
        for t in range(T-2, -1, -1):
            for k in range(K):
                beta_table[t][k] = np.sum([beta_table[t+1][j] * self.trans_probs[k][j] * self.get_emission_prob(j, x[t+1]) for j in range(K)])
            if not np.all(beta_table[t]):
                for k in range(K):
                    beta_table[t][k] = np.sum([beta_table[t+1][j] * self.trans_probs[k][j] * self.get_emission_prob(j, x[t+1]) for j in range(K)]) + MIN_ALPHA_BETA
            beta_table[t,:] /= self.scaling_factors[t+1]
        ''' save results '''        
        self.beta_table = beta_table 

    '''*****************************'''        
    ''' Compute Gamma and Xi Tables '''
    '''*****************************'''

    def compute_xi_table(self, sequence):
        T = len(sequence)
        K = self.num_states
        xi_table = np.zeros((T, K, K))
        for t in range(1, T):
            point = sequence[t]
            sf = self.scaling_factors[t]
            for i in range(K):
                for j in range(K):
                    xi_table[t][i][j] = (1./sf) * self.alpha_table[t-1][i] * self.trans_probs[i][j] * self.get_emission_prob(j, point) * self.beta_table[t][j]        
        self.xi_table = xi_table
        
    def compute_gamma_table(self, sequence):        
        T = len(sequence)
        K = self.num_states
        gamma_table = np.zeros((T, K))
        for t in range(T):
            for k in range(K):
                gamma_table[t][k] = self.alpha_table[t][k] * self.beta_table[t][k]
        assert(self.alpha_table[-1].sum() == gamma_table[-1].sum())        
        self.gamma_table = gamma_table
        self.log_likelihood = np.log(self.scaling_factors).sum()

    '''***************************'''        
    ''' Start Prob Update Methods '''
    '''***************************'''

    def update_start_probs(self):        
        self.start_probs = self.gamma_table[0,:] / np.sum([self.gamma_table[0][j] for j in range(self.num_states)])
        self.start_probs[self.start_probs < MIN_START_PROB] = MIN_START_PROB
        self.start_probs /= self.start_probs.sum()

    '''***************************'''
    ''' Transition Update Methods '''
    '''***************************'''

    def smooth_transition_probs(self):
        for k in range(self.num_states):
            self.trans_probs[k][self.trans_probs[k] < MIN_TRANS_PROB] = MIN_TRANS_PROB
            self.trans_probs[k] /= self.trans_probs[k].sum()

    def update_transition_probs_standard(self, sequence, dummy_param=None):
        T = len(sequence)
        K = self.num_states
        for j in range(K):
            denom = np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)])
            for k in range(K):
                self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] for t in range(1, T)]) / denom
        self.smooth_transition_probs()

    def update_transition_probs_MAP(self, sequence, zeta=1):
        T = len(sequence)
        K = self.num_states
        for j in range(K):
            denom = (zeta - 1) + np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)])
            for k in range(K):
                added_mass = (zeta - 1) if j == k else 0.
                self.trans_probs[j][k] = np.sum([added_mass + self.xi_table[t][j][k] for t in range(1, T)]) / denom
        self.smooth_transition_probs()     

    def update_transition_probs_MAP_scale_free(self, sequence, zeta=1):
        T = len(sequence)
        K = self.num_states
        amp_val = ((T-1)**zeta - 1)
        for j in range(K):
            denom = np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)]) + amp_val
            for k in range(K):
                added_mass = amp_val if j == k else 0.
                self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] + added_mass for t in range(1, T)]) / denom
        self.smooth_transition_probs()        

    def update_transition_probs_inertial_regularized(self, sequence, zeta=1):
        T = len(sequence)
        K = self.num_states
        v = (T-1)**zeta
        amped_vals = [[None for k in range(K)]] + [[(self.gamma_table[t-1][k] - self.xi_table[t][k][k]) * v for k in range(K)] for t in range(1, T)]        
        for j in range(K):
            denom = np.sum([self.xi_table[t][j][i] + (amped_vals[t][i] if i == j else 0.0) for i in range(K) for t in range(1, T)])
            for k in range(K):
                self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] + (amped_vals[t][k] if j == k else 0.0) for t in range(1, T)]) / denom
        self.smooth_transition_probs()                 

    '''************************'''        
    ''' Emission Update Method '''
    '''************************'''

    def update_emission_parameters(self, sequence):
        T = len(sequence)
        K = self.num_states
        x = sequence
        D = x.shape[1]
        emission_means = []
        emission_covariances = []
        for k in range(K):
            denom = self.gamma_table[:,k].sum(axis=0)
            means = np.dot(self.gamma_table[:,k].T, x) / denom
            demeaned = (x - means)
            cov_mat = np.dot(self.gamma_table[:, k] * demeaned.T, demeaned) / denom + MIN_COV_VAL * np.eye(D)
            self.emission_density_objs[k] = multivariate_normal(mean=means, cov=cov_mat)
            emission_means.append(means)
            emission_covariances.append(cov_mat)
        self.prev_emission_means = emission_means
        self.prev_emission_covariances = emission_covariances

    '''************************'''        
    '''    Decode (Viterbi)    '''
    '''************************'''
      
    def decode(self, sequence):
        ''' Implements Viterbi Algorithm '''
        T = len(sequence)
        K = self.num_states
        x = sequence
        V_table = np.zeros((T, K), dtype="float64")
        backpointers = np.zeros((T, K), dtype="float64")
        start_probs = self.start_probs[:]
        if not np.all(start_probs):
            start_probs += 1e-20
            start_probs /= start_probs.sum()
        for k in range(K):
            V_table[0][k] = self.get_log_emission_prob(k, x[0]) + np.log(start_probs[k])
        backpointers[0,:] = -1
        for t in range(1, T):
            for k in range(K):
                log_emiss_p = self.get_log_emission_prob(k, x[t])
                scores = [(V_table[t-1][j] + np.log(self.trans_probs[j][k]) + log_emiss_p,  j) for j in range(K)]
                V_table[t][k], backpointers[t][k] = max(scores)
        state_path = [np.max(backpointers[-1,:])]
        for t in range(backpointers.shape[0]-1, -1, -1):
            state_path.insert(0, backpointers[t][state_path[0]])
        return state_path[1:]

    def incremental_decode(self, sequence, prev_masses=None):
        ''' Implements a modified Viterbi Algorithm '''
        T = len(sequence)
        K = self.num_states
        x = sequence
        V_table = np.zeros((T, K), dtype="float64")
        backpointers = np.zeros((T, K), dtype="float64")        
        if prev_masses is None:
            start_probs = self.start_probs[:]
            if not np.all(start_probs):
                start_probs += 1e-20
                start_probs /= start_probs.sum() 
            for k in range(K):
                V_table[0][k] = self.get_log_emission_prob(k, x[0]) + np.log(start_probs[k])
        else:
            for k in range(K):
                log_emiss_p = self.get_log_emission_prob(k, x[0])
                scores = [prev_masses[j] + np.log(self.trans_probs[j][k]) + log_emiss_p for j in range(K)]
                V_table[0][k] = max(scores)                
        backpointers[0,:] = -1
        for t in range(1, T):
            for k in range(K):
                log_emiss_p = self.get_log_emission_prob(k, x[t])
                scores = [(V_table[t-1][j] + np.log(self.trans_probs[j][k]) + log_emiss_p,  j) for j in range(K)]
                V_table[t][k], backpointers[t][k] = max(scores)
        state_path = [np.max(backpointers[-1,:])]
        for t in range(backpointers.shape[0]-1, -1, -1):
            state_path.insert(0, backpointers[t][state_path[0]])
        return state_path[1:], V_table

    '''************************'''
    '''          Learn         '''
    '''************************'''

    def learn(self, sequence, zeta=None, init=False, only_final_ll=False):
        ''' Runs Baum-Welch to train HMM '''
        K = self.num_states
        epsilon = 0.1
        old_ll = float("-inf")
        iterations = 0
        if init:            
            means, covs = self.get_kmeans_emission_init(sequence)
            self.emission_density_objs = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(K)]            
            start_probs, transition_probs = self.get_random_start_and_trans_probs()
            self.start_probs = start_probs
            self.trans_probs = transition_probs
        while True:
            iterations += 1
            self.forward(sequence)
            self.backward(sequence)
            self.compute_gamma_table(sequence)
            self.compute_xi_table(sequence)
            if not only_final_ll:
                print "Log Likelihood:", self.log_likelihood            
            if abs(self.log_likelihood - old_ll) < epsilon or iterations > MAX_ITER:
                print "Final Log Likelihood:", self.log_likelihood
                return
            else:
                old_ll = self.log_likelihood
            self.update_start_probs()
            self.trans_update_method(sequence, zeta)
            self.update_emission_parameters(sequence)
        ''' Save masses for incrementing '''
        states, masses = self.incremental_decode(sequence)
        self.previous_state_path_masses = masses[-1]

  
    def get_segments(self, state_assignments):
        segments = []
        state = state_assignments[0]
        start = 0
        count = 0
        for i in range(len(state_assignments)):
            if state_assignments[i] == state:
                count += 1
            else:
                segments.append(((start, start+count-1), state))
                start = i
                count = 1
                state = state_assignments[i]
        segments.append(((start, start+count-1), state))                
        return segments
        
    def evaluate_segment_distribution(self, lengths):
        res = GRLC(lengths)
        print "Gini Coeff:", res[1]
        print "Number of segments:", len(lengths)
        print "Avg. Segment Proportion:", np.mean(lengths) / np.sum(lengths)
        return res[1]

    def learn_param_free(self, sequence, param_range=[1,10], only_final_ll=False):
        epsilon = 0.1
        p = np.mean(param_range)
        print "Parameter:", p
        self.learn(sequence, zeta=p, init=True, only_final_ll=only_final_ll)
        hidden_states = self.decode(sequence)
        segments = self.get_segments(hidden_states)
        lengths = sorted([t[0][1] - t[0][0] for t in segments])
        current_gini = self.evaluate_segment_distribution(lengths)
        if abs(param_range[1] - param_range[0]) < epsilon:     
            return p, current_gini        
        if current_gini > MIN_GINI:
            return self.learn_param_free(sequence, [p, param_range[1]], only_final_ll)
        else:
            attempted_p, attempted_gini = self.learn_param_free(sequence, [param_range[0], p], only_final_ll)
            if attempted_gini <= MIN_GINI:
                return attempted_p, attempted_gini
            else:
                return p, current_gini

    def increment(self, x_t, zeta=1.0):
        ''' Initialize. Assumes batch process has already been used to
            learn A^(1), mu^(1), S^(1) and pi.
        '''
        self.t += 1
        K = self.num_states
        self.current_scaling_factor = 1.
        self.current_alpha_vector = np.zeros(K)
        self.current_xi_matrix = np.zeros(shape=(K,K))
        self.current_gamma_vector = np.zeros(K)
        self.current_gamma_sums = np.zeros(K)
        self.current_D_vector = np.zeros(K)
        new_A_matrix = np.zeros(shape=(K,K))
        if self.t == 1:
            print "Start Probs (361):", self.start_probs
            ''' Initialize values for T=1 step '''
            self.prev_alpha_vector = np.zeros(K)
            for j in range(K):
                self.prev_alpha_vector[j] = self.start_probs[j] * self.get_emission_prob(j, x_t) + MIN_ALPHA_BETA
            scaling_factor = self.prev_alpha_vector[:].sum()  
            self.prev_alpha_vector /= scaling_factor
            print self.prev_alpha_vector
            self.prev_gamma_sums = np.zeros(K)
        else: 
            ''' Current alpha-hat values '''
            for j in range(K):
                self.current_alpha_vector[j] = np.sum([self.prev_alpha_vector[i] * self.trans_probs[i][j] for i in range(K)]) * \
                        self.get_emission_prob(j, x_t) + MIN_ALPHA_BETA
            self.current_scaling_factor = self.current_alpha_vector.sum() 
            self.current_alpha_vector /= self.current_scaling_factor
            ''' Current Xi values '''
            for i in range(K):
                for j in range(K):
                    self.current_xi_matrix[i][j] = (self.prev_alpha_vector[i] * self.get_emission_prob(j, x_t) * self.trans_probs[i][j])
                    self.current_xi_matrix[i][j] /= self.current_scaling_factor
            ''' A matrix update '''
            if self.t == 2:
                for i in range(K): 
                    print self.current_xi_matrix[i,:], self.current_xi_matrix[i,:].sum()
                    for j in range(K):
                        self.trans_probs[i][j] = self.current_xi_matrix[i][j] / self.current_xi_matrix[i,:].sum()
                self.smooth_transition_probs()
            else:
                amped_value = ((self.t - 1)**zeta - (self.t - 2)**zeta)
                for i in range(K):
                    self.current_D_vector[i] = amped_value + self.current_xi_matrix[i,:].sum() + self.prev_D_vector[i]
                for i in range(K):
                    for j in range(K):
                        av = amped_value if i == j else 0.
                        last_part = self.prev_D_vector[i] * self.trans_probs[i][j]
                        new_A_matrix[i][j] = (self.current_xi_matrix[i][j] + av + last_part) / self.current_D_vector[i]
                self.trans_probs = new_A_matrix
                self.smooth_transition_probs()
            ''' Current gamma values '''
            self.current_gamma_vector = self.current_alpha_vector[:]
            self.current_gamma_sums = self.current_gamma_vector + self.prev_gamma_sums
            ''' Update emission parameters '''
            coeffs_a = self.prev_gamma_sums / self.current_gamma_sums
            coeffs_b = self.current_gamma_vector / self.current_gamma_sums                        
            emission_means = []
            emission_covariances = []
            for j in range(K):
                current_mean = coeffs_a[j] * self.prev_emission_means[j] + coeffs_b[j] * x_t
                demeaned = (x_t - current_mean)
                cov_mat = coeffs_a[j] * self.prev_emission_covariances[j] + np.dot(coeffs_b[j] * demeaned.T, demeaned) + MIN_COV_VAL * np.eye(x_t.shape[0])
                self.emission_density_objs[j] = multivariate_normal(mean=current_mean, cov=cov_mat)
                emission_means.append(current_mean)
                emission_covariances.append(cov_mat)        
            ''' Save previous values '''
            self.prev_D_vector = self.current_D_vector
            self.prev_alpha_vector = self.current_alpha_vector
            self.prev_gamma_sums = self.current_gamma_sums
            self.prev_emission_means = emission_means
            self.prev_emission_covariances = emission_covariances
        ''' Append data point to data buffer '''
        self.data_buffer.append(x_t)

    def predict_next(self):
        ''' Only predict if we have enough items in buffer '''
        if len(self.data_buffer) < self.window_width:
            return None
        ''' Predict next position in data buffer '''
        sequence = np.array(self.data_buffer)
        prev_masses = self.previous_state_path_masses
        states, masses = self.incremental_decode(sequence, prev_masses)
        #print "States (458):", states
        self.previous_state_path_masses = masses[0]
        ''' Remove predicted data item off of front of data buffer '''
        self.data_buffer.pop(0)
        return (states[0], self.t - (len(self.data_buffer) + 1))

    def increment_and_predict_state(self, x_t, zeta=1.0):
        self.increment(x_t, zeta)
        return self.predict_state()

if __name__ == "__main__":
    '''*****************'''
    ''' Minimal Example '''
    '''*****************'''

    ''' Fake data. Notice each observation is a row, columns are dimensions. 
        Thus, we have 3,000 time steps of 45D data in this fake example. '''
    data = np.random.random((3000, 45))
    data[1500:, :] = 2.

    ''' Regularization modes and parameter '''
    rgzn_modes = InertialHMM.RgznModes()
    zeta = 3.

    ''' Define two-state model, run it on observation data and find 
        maximally likely hidden states, subject to regularization. '''
    K = 2        
    model1 = InertialHMM(K, data, rgzn_modes.INERTIAL)
    model1.learn(data, zeta=zeta)
    predicted_states = model1.decode(data)
