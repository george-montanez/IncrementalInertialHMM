from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

def collapse_state_assignments(state_assignments):
    states = []
    seen = {}
    for s in state_assignments:
        if not s in seen:
            seen[s] = 1
            states.append(s)
    return states            

def reorder_states(state_assignments):
    max_state = max(state_assignments)
    state_order = collapse_state_assignments(state_assignments)
    states = np.array(state_assignments) + max_state
    for i, s in enumerate(state_order):
        states[states==s+max_state] = i
    return states

def visualize_results(data, state_assignments, true_states, num_states, name, save_figs=True):    
    true_states = reorder_states(true_states)
    state_assignments = reorder_states(state_assignments)

    fig = plt.figure(figsize=(14,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[7,1,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)
    axarr=[ax1,ax2,ax3]
    for i in range(num_states):
        idx = (state_assignments == i)
        mod_data = data[:,1].copy()
        for j in range(len(idx)):
            mod_data[j] = mod_data[j] if idx[j] else pl.NaN
        axarr[0].plot(np.arange(len(idx)), mod_data, '-', label="hidden state %d" % (i+1))
        axarr[0].plot(range(len(idx)), np.array(idx, dtype=int)*4 - 2)
    axarr[0].legend(loc=1)
    axarr[0].autoscale_view()
    [label.set_visible(False) for label in axarr[0].get_xticklabels()]
    axarr[1].imshow(np.atleast_2d(state_assignments), interpolation='nearest', cmap=cm.cool, aspect='auto')
    [label.set_visible(False) for label in axarr[1].get_yticklabels()]
    [label.set_visible(False) for label in axarr[1].get_xticklabels()]
    [tick.set_visible(False) for tick in axarr[1].yaxis.get_major_ticks()]
    axarr[1].set_ylabel('Pred.')
    axarr[2].imshow(np.atleast_2d(true_states), interpolation='nearest', cmap=cm.cool, aspect='auto')
    [label.set_visible(False) for label in axarr[2].get_yticklabels()]
    [tick.set_visible(False) for tick in axarr[2].yaxis.get_major_ticks()]
    axarr[2].set_ylabel('True')
    if save_figs:
        pl.savefig('../Proposal/images/%s.pdf' % name, bbox_inches='tight')
        pl.savefig('../Proposal/images/%s.png' % name, dpi=300, bbox_inches='tight')
    pl.show()
    pl.close() 

def visualize_results_no_bars(data, state_assignments, true_states, num_states, name, save_figs=True):    
    true_states = reorder_states(true_states)
    state_assignments = reorder_states(state_assignments)

    fig = plt.figure(figsize=(14,5))
    gs = gridspec.GridSpec(1, 1, height_ratios=[1,])
    ax1 = plt.subplot(gs[0])
    axarr=[ax1,]
    for i in range(num_states):
        idx = (state_assignments == i)
        mod_data = data[:,1].copy()
        for j in range(len(idx)):
            mod_data[j] = mod_data[j] if idx[j] else pl.NaN
        axarr[0].plot(np.arange(len(idx)), mod_data, '-', label="hidden state %d" % (i+1))
        axarr[0].plot(range(len(idx)), np.array(idx, dtype=int)*4 - 2)
    axarr[0].legend(loc=1)
    axarr[0].autoscale_view()
    [label.set_visible(False) for label in axarr[0].get_xticklabels()]
    if save_figs:
        pl.savefig('../Proposal/images/%s.pdf' % name, bbox_inches='tight')
        pl.savefig('../Proposal/images/%s.png' % name, dpi=300, bbox_inches='tight')
    pl.show()
    pl.close()

def visualize_comparison(data, state_assignments_hmm, state_assignments_inert, state_assignments_truth, num_states):    
    fig = pl.figure(figsize=(12,5))
    ax = pl.subplot(411)
    for i in range(num_states):
        idx = (state_assignments == i)
        mod_data = data[:,1].copy()
        for j in range(len(idx)):
            mod_data[j] = mod_data[j] if idx[j] else pl.NaN
        ax.plot(np.arange(len(idx)), mod_data, '-', label="hidden state %d" % (i+1))      
        ax.plot(range(len(idx)), np.array(idx, dtype=int)*4 - 2)        
    ax.legend(loc=1)
    ax.autoscale_view()
    pl.show()

def visualize_dataset(data, true_states, name, save_figs=True): 
    true_states = reorder_states(true_states)
    fig = pl.figure(figsize=(12,4))
    gs = gridspec.GridSpec(2, 2,hspace=0.075,wspace=0.05,width_ratios=[25,1],height_ratios=[7,1])
    ax1 = pl.subplot(gs[0,0])
    im=ax1.imshow(data.T, interpolation='nearest', cmap=cm.spectral,extent=(0,100,0,1.0), aspect='auto')  
    ax1.set_ylabel('Dimensions')
    ax1.set_yticklabels([]) 
    xax1=ax1.get_xaxis()
    xax1.set_ticks([])
    ax3 = pl.subplot(gs[0,1])
    cbar=pl.colorbar(im,cax=ax3)
    ax2 = pl.subplot(gs[1,0])
    ax2.imshow(np.atleast_2d(true_states), interpolation=None, cmap=cm.winter, aspect='auto')
    ax2.get_yaxis().set_ticks([])
    ax2.set_ylabel('State')
    if save_figs or True:
        pl.savefig('../Proposal/images/%s.pdf' % name, bbox_inches='tight')
        pl.savefig('../Proposal/images/%s.png' % name, dpi=300, bbox_inches='tight')
    pl.show()
