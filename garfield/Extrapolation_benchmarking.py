#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 16:35:00 2024

@author: maltheraschke
"""

import os, sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c_light
from scipy.signal import find_peaks
from wakis import WakeSolver

sys.path.append('../garfield/')
from Genetic_algorithm import *
from Minimization_functions import *
from Resonator_formula import *
from framework import GeneticAlgorithm


def x_attenuation_data(wake_data, energy_data, attenuation):
    minus_x_db_time = energy_data[np.argmax(energy_data[30:, 1] < attenuation) + 30, 0]
    return np.argmax(wake_data[:, 0] > (minus_x_db_time*1e-9 * c_light * 1e2))

def impedance_at_attenuations(attenuations_list, wake_data, energy_data, peak_threshold=1e3, sigmaz=0.1):
    row_indices = {}
    for attenuation in attenuations_list:
        row_indices[f'minus{int(abs(attenuation))}dB'] = x_attenuation_data(wake_data, energy_data, attenuation)

    impedances_attenuations = row_indices.copy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))

    for idx, (key, value) in enumerate(row_indices.items()):
        if key != 'key':  # Skip the 'key' entry
            row = idx // 4
            col = idx % 4
            wake = WakeSolver(q=1e-9, sigmaz=sigmaz)
            wake.WP = wake_data[:value, 1]
            wake.s = wake_data[:value, 0] / 100  # Converting from cm to m
            wake.calc_lambdas_analytic()
            wake.calc_long_Z(samples=1001)
            peaks, peaks_height = find_peaks(np.abs(wake.Z), height=peak_threshold, threshold=None)

            axes[row, col].plot(wake.f, np.abs(wake.Z))
            axes[row, col].set_title(f'{key} at wakelength: {wake_data[value, 0]/100:.1f}m')
            axes[row, col].set_xlabel('Frequency [Hz]')
            axes[row, col].set_ylabel('Impedance [Ohm]')
            for peak, height in zip(peaks, peaks_height['peak_heights']):
                axes[row, col].text(wake.f[peak], np.abs(wake.Z)[peak], f'{wake.f[peak]/1e9:.2f} GHz\n{height:.2f}', fontsize=9)
            axes[row, col].grid(True)       

            impedances_attenuations.update({key: wake.Z})

    fig.tight_layout()
    plt.show()

    return impedances_attenuations, wake.f, row_indices

def run_genetic_algoritms(impedances_attenuations_dict, attenuations_indicies_dict, frequency, wake_data, time, wake, Nres, parameterBounds, maxiter=30000, tol=0.001):
    results = {}
    for key, impedance in impedances_attenuations_dict.items():
        n_Resonator_longitudinal_partial_decay_imp = partial(n_Resonator_longitudinal_imp, wake_length=wake_data[attenuations_indicies_dict[key], 0]/100)
        GA_model_scipy = GeneticAlgorithm(frequency, 
                                          impedance, 
                                          time, 
                                          wake, 
                                          N_resonators=Nres,
                                          parameterBounds=parameterBounds,
                                          minimizationFunction=sumOfSquaredError,
                                          fitFunction=n_Resonator_longitudinal_partial_decay_imp
                                         )
        GA_model_scipy.run_geneticAlgorithm(maxiter=maxiter, 
                                            popsize=150, 
                                            tol=tol,
                                            mutation=(0.1, 0.5), 
                                            crossover_rate=0.8
                                           )
        results[key] = GA_model_scipy
    print(GA_model_scipy.warning)
    return results

def plot_GA_results(GA_results):

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, (key, value) in enumerate(GA_results.items()):
        if key != 'key':  # Skip the 'key' entry
            row = idx // 4
            col = idx % 4

            axes[row, col].plot(GA_results[key].frequency_data, GA_results[key].impedance_data, "black", label='CST data')
            axes[row, col].plot(GA_results[key].frequency_data, GA_results[key].fitFunction(GA_results[key].frequency_data, dict(enumerate(GA_results[key].minimizationParameters.reshape(-1, 3)))).real,
                lw = 3, linestyle='--', label='Scipy', alpha=0.7)
            axes[row, col].set_title(f'{key}')
            axes[row, col].set_xlabel('Frequency [Hz]')
            axes[row, col].set_ylabel('Impedance [Ohm]')
    fig.tight_layout()
    plt.show()

def timeframe_for_extrapolation(results, end_time):
    """The time of 210m wakelength is estimated by envelope exponential decay fitting."""

    return np.linspace(list(results.values())[0].time_data[0], end_time, int(np.round(len(list(results.values())[0].time_data)*(end_time/(list(results.values())[0].time_data[-1])))))

def compute_longitudinal_wake_function(results, new_time_extrapolate):
    wake_extrapolations = {}

    for idx, (key, value) in enumerate(results.items()):
        if key != 'key':  # Skip the 'key' entry
            wake_extrapolated = n_Resonator_longitudinal_wake(new_time_extrapolate, dict(enumerate(results[key].minimizationParameters.reshape(-1, 3))))*1e-13
            wake_extrapolated = np.column_stack((new_time_extrapolate, wake_extrapolated))
            wake_extrapolations[key] = wake_extrapolated
    return wake_extrapolations

def compute_impedance_wakis(wake_data, q_val = 1e-9, sigmaz_val = 0.1):
    #instantiating a class. An instance of the WakeSolver class.
    wake_fully_simulated = WakeSolver(q = q_val, sigmaz = sigmaz_val)

    wake_fully_simulated.WP = wake_data[:,1]

    wake_fully_simulated.s = wake_data[:,0]/100

    wake_fully_simulated.calc_lambdas_analytic()

    wake_fully_simulated.calc_long_Z(samples=1001)

    peaks_fs, peaks_height_fs = find_peaks(np.abs(wake_fully_simulated.Z), height=2e3, threshold=None)

    return wake_fully_simulated, peaks_fs, peaks_height_fs

def compare_extrapolation_results(wake_frequencies, wake_extrapolations, new_time_extrapolate, data_wake, charge_dist, wake_fully_simulated, peaks_fs, peaks_height_fs):
    # Plotting the impedance of the fully simulated data
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    peak_info = ', '.join([f'{wake_frequencies[peak]/1e6:.2f} MHz: {height:.2f}' for peak, height in zip(peaks_fs, peaks_height_fs['peak_heights'])])
    st = fig.suptitle(f'Peaks at {peak_info}', fontsize=20)

    impedance_results = {}

    for idx, (key, value) in enumerate(wake_extrapolations.items()):
        if key != 'key':  # Skip the 'key' entry
            row = idx // 4
            col = idx % 4
            #instantiating a class. An instance of the WakeSolver class.
            wake = WakeSolver(q = 1e-9, sigmaz = 0.1)

            #setting the wake potential and time data

            wake.WP = value[:, 1]

            wake.s = new_time_extrapolate*c_light

            wake.lambdas = np.interp(data_wake[:, 0], charge_dist[:, 0], charge_dist[:, 1] / 1e-9 *1e3)

            # Computing the impedance using the WakeSolver class and plotting the result

            wake.calc_long_Z(samples = 1001)
            peaks, peaks_height = find_peaks(np.abs(wake.Z), height=2e3, threshold=None)
            
            axes[row, col].plot(wake_fully_simulated.f, np.abs(wake_fully_simulated.Z), color='black')
            axes[row, col].plot(wake.f, np.abs(wake.Z), linestyle='--')
            axes[row, col].set_title(f'{key}')
            axes[row, col].set_xlabel('Frequency [Hz]')
            axes[row, col].set_ylabel('Impedance [Ohm]')
            for peak, height in zip(peaks, peaks_height['peak_heights']):
                axes[row, col].text(wake.f[peak], np.abs(wake.Z)[peak], f'{wake.f[peak]/1e6:.2f} MHz\n{height:.2f}', fontsize=9)
            axes[row, col].grid(True)

            mse = np.mean((np.abs(wake_fully_simulated.Z) - np.abs(wake.Z))**2)
            axes[row, col].text(.01, .99, f'MSE: {mse:.2f}', horizontalalignment='left', verticalalignment='top', color='purple', transform=axes[row, col].transAxes)
            delta_fundamental = peaks_height_fs["peak_heights"][0] - peaks_height["peak_heights"][0]
            axes[row, col].text(.01, .90, f'$\Delta Fundamental$: {delta_fundamental:.2f} ({(np.abs(delta_fundamental/peaks_height_fs["peak_heights"][0]))*100:.2f})%', horizontalalignment='left', verticalalignment='top', color='purple', transform=axes[row, col].transAxes)
            delta_harmonic = peaks_height_fs["peak_heights"][1] - peaks_height["peak_heights"][1]
            axes[row, col].text(.01, .86, f'$\Delta harmonic$: {delta_harmonic:.2f} ({(np.abs(delta_harmonic/peaks_height_fs["peak_heights"][1]))*100:.2f})%', horizontalalignment='left', verticalalignment='top', color='purple', transform=axes[row, col].transAxes)

    fig.tight_layout()
    plt.show()