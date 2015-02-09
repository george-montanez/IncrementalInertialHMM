from __future__ import division
import numpy as np
from scipy import stats

def sig_marker(marks):
    return "{0:s}\phantom{{{1:*<{2}}}}".format(marks, "", 3 - len(marks))

def statistically_significant(sample1, sample2):
    """ Does paired t-test for samples. Returns p-value and (0.05, 0.01, < 0.001)
        significance results for the difference of sample means, for each metric. 
    """
    tstats, pvals = stats.ttest_ind(sample1, sample2, equal_var=False)    
    return [(pval, sig_marker("*" * (pval < 0.05) + "*" *(pval < 0.01) + "*" * (pval < 0.001))) for pval in pvals]

def get_results_arrays(results_dir):
    results = {}
    for key in ('INERTIAL', 'MAP', 'STANDARD', 'STICKY'):
        results[key] = np.loadtxt('%squant_analysis_%s.txt' % (results_dir, key))
    return results

""" ************** """
""" Main Functions """
""" ************** """

def main(results_dir):
    results_arrays = get_results_arrays(results_dir)
    baseline = results_arrays['MAP']
    for key in ('STANDARD', 'INERTIAL', 'STICKY'):
        pvals = statistically_significant(baseline, results_arrays[key])        
        print key
        for tup, label in zip(pvals, ('ACC','SNR','ASR','DIF','PER','VOI',)):
            print label, "\t", tup
        print "*" * 100            

if __name__ == "__main__":
    print "=" * 100
    print "SYNTHETIC RESULTS"
    print "=" * 100
    main('../RESULTS-ALL-SYNTH/')
    print "=" * 100
    print "REAL DATA RESULTS"
    print "=" * 100
    main('../RESULTS-ALL-REAL/')
