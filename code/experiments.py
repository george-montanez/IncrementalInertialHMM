from __future__ import division
import numpy as np
from visualize import *
from InertialHMM import InertialHMM
from evaluation import *
import os

DATA_DIR = "../data/LongAccelerometerSeries/"
RD_DATA_FILE_PATTERN = "./%d_accelerometer_data.txt"
RD_LABEL_FILE_PATTERN = "./%d_accelerometer_labels.txt"
SD_DATA_FILE_PATTERN = "./%d_synth_data.txt"
SD_LABEL_FILE_PATTERN = "./%d_synth_labels.txt"
rgzn_modes = InertialHMM.RgznModes()

def learn_initial_model(data, K):
    model = InertialHMM(K, data, rgzn_modes.MAP_SCALE_FREE)
    final_param, gini_coeff = model.learn_param_free(data, param_range=[0.,100.])
    print "Final Param", final_param
    model.learn(data, zeta=final_param, init=True)
    return model, final_param

def learn_initial_model_quick(data, K):
    final_param = 10.
    model = InertialHMM(K, data, rgzn_modes.MAP_SCALE_FREE)    
    model.learn(data, zeta=final_param, init=True)
    return model, final_param

def do_quantitative(results_dir = "../results/"): 
    evaluation_dir = results_dir + "/evaluation/"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    results = []
    suffix = "INC_INERTIAL"
    i = 0
    INIT_DATA_LEN = 10000
    while i < 2:
        #try:
        if True:
            print "Attempting:", i
            data = np.loadtxt(DATA_DIR + RD_DATA_FILE_PATTERN % i, delimiter=" ")
            true_states = np.loadtxt(DATA_DIR + RD_LABEL_FILE_PATTERN % i, dtype="int8", delimiter=" ")[:INIT_DATA_LEN]
            true_states -= np.min(true_states)
            print len(true_states)
            K = len(set(true_states))
            model, zeta = learn_initial_model(data[-INIT_DATA_LEN:], K)
            predicted_states = []
            while len(predicted_states) < len(data) - INIT_DATA_LEN:
                model.increment(data[model.t], zeta)
                res = model.predict_next()
                if not res is None:
                    predicted_states.append(res[0])
                    print res[0], true_states[model.t-1]
            #visualize_results(data, np.array(predicted_states), true_states, K, 'MAP_PARAM_FREE_results_%s_%0.2f_%d_states' % ('fake', 5., K), save_figs=False)
            pt = np.hstack((np.array(predicted_states).reshape((-1,1)), np.array(true_states).reshape((-1,1))))
            np.savetxt(evaluation_dir + "./%s_%d_indv_predictions.txt" % (suffix, i), pt, fmt="%d")
            ents = entropies(predicted_states, true_states)
            normed_voi = (ents[0] + ents[1] - 2 * ents[3]) / ents[2]
            res = (max_correct(predicted_states, true_states), segment_measurements(predicted_states, true_states), normed_voi)
            results.append(res)
            print i, res, len(segment_func(true_states))
            i += 1
        else:
        #except:        
            print "\n************* Example %d Failed *************" % i  
    accuracies = [r[0][1] for r in results]
    actual_ratios = [r[1][0] for r in results]
    divergence_ratios = [r[1][1] for r in results]
    diff_in_num_of_segments = [r[1][2] for r in results]
    perfect_segmentations = [r[0][2] for r in results]
    vois = [r[2] for r in results]
    print "=" * 40
    print "=" * 40
    print "Avg. Accuracy", np.mean(accuracies)
    print "Avg. Segment Number Ratio", np.mean(actual_ratios)
    print "Avg. Segment Number Divergence Ratio", np.mean(divergence_ratios)
    print "Avg. Number of Segments Difference", np.mean(diff_in_num_of_segments)
    print "Total Number of Perfect Segmentations", np.sum(perfect_segmentations), "of", i + 1
    print "Avg. Normed Variation of Information", np.mean(vois)
    print "=" * 40
    print "=" * 40
    out = open(results_dir + "./results_%s.txt" % suffix, "w")
    output = ("Avg. Accuracy %.3f \n" + \
              "Avg. Segment Number Ratio %.2f \n" + \
              "Avg. Segment Number Divergence Ratio %.2f\n" + \
              "Avg. Number of Segments Difference %.2f\n" + \
              "Total Number of Perfect Segmentations %d \n" + \
              "Avg. Variation of Information %.2f") % (np.mean(accuracies),
                                                     np.mean(actual_ratios),
                                                     np.mean(divergence_ratios),
                                                     np.mean(diff_in_num_of_segments), 
                                                     np.sum(perfect_segmentations),
                                                     np.mean(vois))
    out.write(output)
    all_results = np.vstack((accuracies,
                             actual_ratios,
                             divergence_ratios,
                             diff_in_num_of_segments,
                             perfect_segmentations,
                             vois)).T
    np.savetxt(results_dir + "./quant_analysis_%s.txt" % suffix, all_results)
    
if __name__ == "__main__":
    do_quantitative(results_dir='../results/IncrementalMAP/')

