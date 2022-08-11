#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:13:33 2018

@author: Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)
"""
import pandas as pd
from bokeh.layouts import layout, gridplot
from bokeh.models import Range1d, LinearAxis, Legend, Div
from bokeh.palettes import small_palettes
from bokeh.plotting import figure
from lmfit import minimize, report_fit, fit_report
import seaborn as sns
import numpy as np

from .circuits import CIRCUIT_DICT, leastsq_errorfunc
from .PyEIS_Data_extraction import *
from .PyEIS_Lin_KK import *
from .PyEIS_Advanced_tools import *

pd.options.mode.chained_assignment = None

# mpl.rc('mathtext', fontset='stixsans', default='regular')
# mpl.rcParams.update({'axes.labelsize': 10})
# mpl.rc('xtick', labelsize=10)
# mpl.rc('ytick', labelsize=10)
# mpl.rc('legend', fontsize=10)


# Frequency generator
def freq_gen(f_start, f_stop, pts_decade=7):
    """
    Frequency Generator with logspaced freqencies

    Inputs
    ----------
    f_start = frequency start [Hz]
    f_stop = frequency stop [Hz]
    pts_decade = Points/decade, default 7 [-]

    Output
    ----------
    [0] = frequency range [Hz]
    [1] = Angular frequency range [1/s]
    """
    f_decades = np.log10(f_start) - np.log10(f_stop)
    f_range = np.logspace(np.log10(f_start), np.log10(f_stop),
                          num=int(np.around(pts_decade*f_decades)), endpoint=True)
    w_range = 2 * np.pi * f_range
    return f_range, w_range


# Fitting Class
class EIS_exp:
    """
    This class is used to plot and/or analyze experimental impedance data. The class has three
    major functions:
        - EIS_plot()
        - Lin_KK()
        - EIS_fit()

    - EIS_plot() is used to plot experimental data with or without fit
    - Lin_KK() performs a linear Kramers-Kronig analysis of the experimental data set.
    - EIS_fit() performs complex non-linear least-squares fitting of the experimental data to an
     equivalent circuit

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Inputs
    -----------
        - path: path of datafile(s) as a string
        - data: datafile(s) including extension, e.g. ['EIS_data1', 'EIS_data2']
        - cycle: Specific cycle numbers can be extracted using the cycle function. Default is
        'none', which includes all cycle numbers.
        Specific cycles can be extracted using this parameter, insert cycle numbers in brackets,
         e.g. cycle number 1,4, and 6 are wanted. cycle=[1,4,6]
        - mask: ['high frequency' , 'low frequency'], if only a high- or low-frequency is desired
        use 'none' for the other, e.g. maks=[10**4,'none']
    """
    def __init__(self, path, data, cycle='off', mask=['none','none']):
        self.df_raw0 = []
        self.cycleno = []
        for j in range(len(data)):
            if data[j].find(".mpt") != -1: #file is a .mpt file
                self.df_raw0.append(extract_mpt(path=path, EIS_name=data[j])) #reads all datafiles
            elif data[j].find(".DTA") != -1: #file is a .dta file
                self.df_raw0.append(extract_dta(path=path, EIS_name=data[j])) #reads all datafiles
            elif data[j].find(".z") != -1: #file is a .z file
                self.df_raw0.append(extract_solar(path=path, EIS_name=data[j])) #reads all datafiles
            else:
                print('Data file(s) could not be identified')

            self.cycleno.append(self.df_raw0[j].cycle_number)
            if np.min(self.cycleno[j]) <= np.max(self.cycleno[j-1]):
                if j > 0: #corrects cycle_number except for the first data file
                    self.df_raw0[j].update({'cycle_number': self.cycleno[j]+np.max(self.cycleno[j-1])}) #corrects cycle number
#            else:
#                print('__init__ Error (#1)')

        #currently need to append a cycle_number coloumn to gamry files

        # adds individual dataframes into one
        self.df_raw = pd.concat([df for df in self.df_raw0], axis=0)
        # creates a new coloumn with the angular frequency
        self.df_raw = self.df_raw.assign(w=2*np.pi*self.df_raw.f)

        #Masking data to each cycle
        self.df_pre = []
        self.df_limited = []
        self.df_limited2 = []
        self.df = []
        if mask == ['none','none'] and cycle == 'off':
            for i in range(len(self.df_raw.cycle_number.unique())): #includes all data
                self.df.append(self.df_raw[self.df_raw.cycle_number == self.df_raw.cycle_number.unique()[i]])
        elif mask == ['none','none'] and cycle != 'off':
            for i in range(len(cycle)):
                self.df.append(self.df_raw[self.df_raw.cycle_number == cycle[i]]) #extracting dataframe for each cycle
        elif mask[0] != 'none' and mask[1] == 'none' and cycle == 'off':
            self.df_pre = self.df_raw.mask(self.df_raw.f > mask[0])
            self.df_pre.dropna(how='all', inplace=True)
            for i in range(len(self.df_pre.cycle_number.unique())): #Appending data based on cycle number
                self.df.append(self.df_pre[self.df_pre.cycle_number == self.df_pre.cycle_number.unique()[i]])
        elif mask[0] != 'none' and mask[1] == 'none' and cycle != 'off': # or [i for i, e in enumerate(mask) if e == 'none'] == [0]
            self.df_limited = self.df_raw.mask(self.df_raw.f > mask[0])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited.cycle_number == cycle[i]])
        elif mask[0] == 'none' and mask[1] != 'none' and cycle == 'off':
            self.df_pre = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_pre.dropna(how='all', inplace=True)
            for i in range(len(self.df_raw.cycle_number.unique())): #includes all data
                self.df.append(self.df_pre[self.df_pre.cycle_number == self.df_pre.cycle_number.unique()[i]])
        elif mask[0] == 'none' and mask[1] != 'none' and cycle != 'off':
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited.cycle_number == cycle[i]])
        elif mask[0] != 'none' and mask[1] != 'none' and cycle != 'off':
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_limited2 = self.df_limited.mask(self.df_raw.f > mask[0])
            for i in range(len(cycle)):
                self.df.append(self.df_limited[self.df_limited2.cycle_number == cycle[i]])
        elif mask[0] != 'none' and mask[1] != 'none' and cycle == 'off':
            self.df_limited = self.df_raw.mask(self.df_raw.f < mask[1])
            self.df_limited2 = self.df_limited.mask(self.df_raw.f > mask[0])
            for i in range(len(self.df_raw.cycle_number.unique())):
                self.df.append(self.df_limited[self.df_limited2.cycle_number == self.df_raw.cycle_number.unique()[i]])
        else:
            print('__init__ error (#2)')

        # other attrs
        self.fit = []
        self.circuit_fit = []
        self.init_fit = []
        self.fit_reports = []

    def Lin_KK(self,
               num_RC='auto',
               legend='on',
               plot='residuals',
               bode='off',
               nyq_xlim='none',
               nyq_ylim='none',
               weight_func='Boukamp',
               savefig='none'):
        """
        Plots the Linear Kramers-Kronig (KK) Validity Test
        The script is based on Boukamp and Schōnleber et al.'s papers for fitting the resistances of multiple -(RC)- circuits
        to the data. A data quality analysis can hereby be made on the basis of the relative residuals

        Ref.:
            - Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
            - Boukamp, B.A. J. Electrochem. Soc., 142, 6, 1885-1894

        The function performs the KK analysis and as default the relative residuals in each subplot

        Note, that weigh_func should be equal to 'Boukamp'.

        Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

        Optional Inputs
        -----------------
        - num_RC:
            - 'auto' applies an automatic algorithm developed by Schōnleber, M. et al. Electrochimica Acta 131 (2014) 20-27
            that ensures no under- or over-fitting occurs
            - can be hardwired by inserting any number (RC-elements/decade)

        - plot:
            - 'residuals' = plots the relative residuals in subplots correspoding to the cycle numbers picked
            - 'w_data' = plots the relative residuals with the experimental data, in Nyquist and bode plot if desired, see 'bode =' in description

        - nyq_xlim/nyq_xlim: Change the x/y-axis limits on nyquist plot, if not equal to 'none' state [min,max] value

        - legend:
            - 'on' = displays cycle number
            - 'potential' = displays average potential which the spectra was measured at
            - 'off' = off

        bode = Plots Bode Plot - options:
            'on' = re, im vs. log(freq)
            'log' = log(re, im) vs. log(freq)

            're' = re vs. log(freq)
            'log_re' = log(re) vs. log(freq)

            'im' = im vs. log(freq)
            'log_im' = log(im) vs. log(freq)
        """
        if num_RC == 'auto':
            print('cycle || No. RC-elements ||   u')
            self.decade = []
            self.Rparam = []
            self.t_const = []
            self.Lin_KK_Fit = []
            self.R_names = []
            self.KK_R0 = []
            self.KK_R = []
            self.number_RC = []
            self.number_RC_sort = []

            self.KK_u = []
            self.KK_Rgreater = []
            self.KK_Rminor = []
            M = 2
            for i in range(len(self.df)):
                #determine the number of RC circuits based on the number of decades measured and num_RC
                self.decade.append(np.log10(np.max(self.df[i].f))-np.log10(np.min(self.df[i].f)))
                self.number_RC.append(M)
                self.number_RC_sort.append(M)  # needed for self.KK_R
                #Creates intial guesses for R's
                self.Rparam.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[0])
                #Creates time constants values for self.number_RC -(RC)- circuits
                self.t_const.append(KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC[i])))

                self.Lin_KK_Fit.append(minimize(KK_errorfunc, self.Rparam[i], method='leastsq',
                                                args=(self.df[i].w.values,
                                                      self.df[i].re.values,
                                                      self.df[i].im.values,
                                                      self.number_RC[i],
                                                      weight_func,
                                                      self.t_const[i]))) # maxfev=99
                self.R_names.append(KK_Rnam_val(re=self.df[i].re,
                                                re_start=self.df[i].re.idxmin(),
                                                num_RC=int(self.number_RC[i]))[1]) # creates R names
                for j in range(len(self.R_names[i])):
                    self.KK_R0.append(self.Lin_KK_Fit[i].params.get(self.R_names[i][j]).value)
            self.number_RC_sort.insert(0,0) #needed for self.KK_R
            for i in range(len(self.df)):
                self.KK_R.append(self.KK_R0[int(np.cumsum(self.number_RC_sort)[i]):int(np.cumsum(self.number_RC_sort)[i+1])]) #assigns resistances from each spectra to their respective df
                self.KK_Rgreater.append(np.where(np.array(self.KK_R)[i] >= 0, np.array(self.KK_R)[i], 0) )
                self.KK_Rminor.append(np.where(np.array(self.KK_R)[i] < 0, np.array(self.KK_R)[i], 0) )
                self.KK_u.append(1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i]))))

            for i in range(len(self.df)):
                while self.KK_u[i] <= 0.75 or self.KK_u[i] >= 0.88:
                    self.number_RC_sort0 = []
                    self.KK_R_lim = []
                    self.number_RC[i] = self.number_RC[i] + 1
                    self.number_RC_sort0.append(self.number_RC)
                    self.number_RC_sort = np.insert(self.number_RC_sort0, 0,0)
                    #Creates intial guesses for R's
                    self.Rparam[i] = KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[0]
                    #Creates time constants values for self.number_RC -(RC)- circuits
                    self.t_const[i] = KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC[i]))
                    self.Lin_KK_Fit[i] = minimize(KK_errorfunc, self.Rparam[i], method='leastsq',
                                                  args=(self.df[i].w.values,
                                                        self.df[i].re.values,
                                                        self.df[i].im.values,
                                                        int(self.number_RC[i]),
                                                        weight_func,
                                                        self.t_const[i]) ) #maxfev=99
                    self.R_names[i] = KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(), num_RC=int(self.number_RC[i]))[1] #creates R names
                    self.KK_R0 = np.delete(np.array(self.KK_R0), np.s_[0:len(self.KK_R0)])
                    self.KK_R0 = []
                    for q in range(len(self.df)):
                        for j in range(len(self.R_names[q])):
                            self.KK_R0.append(self.Lin_KK_Fit[q].params.get(self.R_names[q][j]).value)
                    self.KK_R_lim = np.cumsum(self.number_RC_sort) #used for KK_R[i]

                    #assigns resistances from each spectra to their respective df
                    self.KK_R[i] = self.KK_R0[self.KK_R_lim[i]:self.KK_R_lim[i+1]]
                    self.KK_Rgreater[i] = np.where(np.array(self.KK_R[i]) >= 0, np.array(self.KK_R[i]), 0)
                    self.KK_Rminor[i] = np.where(np.array(self.KK_R[i]) < 0, np.array(self.KK_R[i]), 0)
                    self.KK_u[i] = 1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i])))
                else:
                    print('['+str(i+1)+']'+'            '+str(self.number_RC[i]),
                          '           '+str(np.round(self.KK_u[i],2)))

        elif num_RC != 'auto': #hardwired number of RC-elements/decade
            print('cycle ||   u')
            self.decade = []
            self.number_RC0 = []
            self.number_RC = []
            self.Rparam = []
            self.t_const = []
            self.Lin_KK_Fit = []
            self.R_names = []
            self.KK_R0 = []
            self.KK_R = []
            for i in range(len(self.df)):
                #determine the number of RC circuits based on the number of decades measured and num_RC
                self.decade.append(np.log10(np.max(self.df[i].f))-np.log10(np.min(self.df[i].f)))
                self.number_RC0.append(np.round(num_RC * self.decade[i]))
                self.number_RC.append(np.round(num_RC * self.decade[i])) #Creats the the number of -(RC)- circuits
                #Creates intial guesses for R's
                self.Rparam.append(KK_Rnam_val(re=self.df[i].re, re_start=self.df[i].re.idxmin(),
                                               num_RC=int(self.number_RC0[i]))[0])
                #Creates time constants values for self.number_RC -(RC)- circuits
                self.t_const.append(KK_timeconst(w=self.df[i].w, num_RC=int(self.number_RC0[i])))
                self.Lin_KK_Fit.append(minimize(KK_errorfunc, self.Rparam[i], method='leastsq',
                                                args=(self.df[i].w.values,
                                                      self.df[i].re.values,
                                                      self.df[i].im.values,
                                                      int(self.number_RC0[i]),
                                                      weight_func,
                                                      self.t_const[i]))) #maxfev=99
                #creates R names
                self.R_names.append(KK_Rnam_val(re=self.df[i].re,
                                                re_start=self.df[i].re.idxmin(),
                                                num_RC=int(self.number_RC0[i]))[1])
                for j in range(len(self.R_names[i])):
                    self.KK_R0.append(self.Lin_KK_Fit[i].params.get(self.R_names[i][j]).value)
            self.number_RC0.insert(0,0)

    #        print(report_fit(self.Lin_KK_Fit[i])) # prints fitting report

            self.KK_circuit_fit = []
            self.KK_rr_re = []
            self.KK_rr_im = []
            self.KK_Rgreater = []
            self.KK_Rminor = []
            self.KK_u = []
            for i in range(len(self.df)):
                #assigns resistances from each spectra to their respective df
                self.KK_R.append(self.KK_R0[int(np.cumsum(self.number_RC0)[i]):int(np.cumsum(self.number_RC0)[i+1])])
                self.KK_Rx = np.array(self.KK_R)
                self.KK_Rgreater.append(np.where(self.KK_Rx[i] >= 0, self.KK_Rx[i], 0) )
                self.KK_Rminor.append(np.where(self.KK_Rx[i] < 0, self.KK_Rx[i], 0) )
                self.KK_u.append(1-(np.abs(np.sum(self.KK_Rminor[i]))/np.abs(np.sum(self.KK_Rgreater[i])))) #currently gives incorrect values
                print('['+str(i+1)+']'+'       '+str(np.round(self.KK_u[i],2)))
        else:
            print('num_RC incorrectly defined')

        self.KK_circuit_fit = []
        self.KK_rr_re = []
        self.KK_rr_im = []
        for i in range(len(self.df)):
            self.KK_circuit_fit.append(KK_RC(w=self.df[i].w,
                                             Rs=self.Lin_KK_Fit[i].params.get('Rs').value,
                                             R_values=self.KK_R[i],
                                             t_values=self.t_const[i],
                                             num_RC=int(self.number_RC[i])))

            # relative residuals for the real part
            self.KK_rr_re.append(residual_real(re=self.df[i].re,
                                               fit_re=self.KK_circuit_fit[i].real,
                                               fit_im=-self.KK_circuit_fit[i].imag))
            # relative residuals for the imag part
            self.KK_rr_im.append(residual_imag(im=self.df[i].im,
                                               fit_re=self.KK_circuit_fit[i].real,
                                               fit_im=-self.KK_circuit_fit[i].imag))

        ### Plotting Linear_kk results
        ##
        #
        ### Label functions
        self.label_re_1 = []
        self.label_im_1 = []
        self.label_cycleno = []
        if legend == 'on':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' (#"+str(i+1)+")")
                self.label_im_1.append("Z'' (#"+str(i+1)+")")
                self.label_cycleno.append('#'+str(i+1))
        elif legend == 'potential':
            for i in range(len(self.df)):
                self.label_re_1.append("Z' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_im_1.append("Z'' ("+str(np.round(np.average(self.df[i].E_avg), 2))+' V)')
                self.label_cycleno.append(str(np.round(np.average(self.df[i].E_avg), 2))+' V')

        if plot == 'w_data':
            fig = figure(figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(311, aspect='equal')
            ax1 = fig.add_subplot(312)
            ax2 = fig.add_subplot(313)

            colors = sns.color_palette("colorblind", n_colors=len(self.df))
            colors_real = sns.color_palette("Blues", n_colors=len(self.df)+2)
            colors_imag = sns.color_palette("Oranges", n_colors=len(self.df)+2)

            ### Nyquist Plot
            for i in range(len(self.df)):
                ax.plot(self.df[i].re, self.df[i].im, marker='o', ms=4, lw=2, color=colors[i],
                        ls='-', alpha=.7, label=self.label_cycleno[i])

            ### Bode Plot
            if bode == 'on':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i+1],
                             marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_re_1[i])
                    ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i+1],
                             marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_im_1[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("Z', -Z'' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            elif bode == 're':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].re, color=colors_real[i+1],
                             marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("Z' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            elif bode == 'log_re':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i+1],
                             marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(Z') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            elif bode == 'im':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), self.df[i].im, color=colors_imag[i+1],
                             marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("-Z'' [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            elif bode == 'log_im':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i+1],
                             marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_cycleno[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(-Z'') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            elif bode == 'log':
                for i in range(len(self.df)):
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].re), color=colors_real[i+1],
                             marker='D', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_re_1[i])
                    ax1.plot(np.log10(self.df[i].f), np.log10(self.df[i].im), color=colors_imag[i+1],
                             marker='s', ms=3, lw=2.25, ls='-', alpha=.7, label=self.label_im_1[i])
                    ax1.set_xlabel("log(f) [Hz]")
                    ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
                    if legend == 'on' or legend == 'potential':
                        ax1.legend(loc='best',  frameon=False)

            # Kramers-Kronig Relative Residuals
            for i in range(len(self.df)):
                ax2.plot(np.log10(self.df[i].f), self.KK_rr_re[i]*100, color=colors_real[i+1],
                         marker='D', ls='--', ms=6, alpha=.7, label=self.label_re_1[i])
                ax2.plot(np.log10(self.df[i].f), self.KK_rr_im[i]*100, color=colors_imag[i+1],
                         marker='s', ls='--', ms=6, alpha=.7, label=self.label_im_1[i])
                ax2.set_xlabel("log(f) [Hz]")
                ax2.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
            ax2.axhline(0, ls='--', c='k', alpha=.5)

            # Setting ylims and write 'KK-Test' on RR subplot
            self.KK_rr_im_min = []
            self.KK_rr_im_max = []
            self.KK_rr_re_min = []
            self.KK_rr_re_max = []
            for i in range(len(self.df)):
                self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
            if np.min(self.KK_rr_im_min) > np.min(self.KK_rr_re_min):
                ax2.set_ylim(np.min(self.KK_rr_re_min)*100*1.5, np.max(np.abs(self.KK_rr_re_min))*100*1.5)
                ax2.annotate('Lin-KK',
                             xy=[np.min(np.log10(self.df[0].f)), np.max(self.KK_rr_re_max)*100*.9],
                             color='k', fontweight='bold')
            elif np.min(self.KK_rr_im_min) < np.min(self.KK_rr_re_min):
                ax2.set_ylim(np.min(self.KK_rr_im_min)*100*1.5, np.max(self.KK_rr_im_max)*100*1.5)
                ax2.annotate('Lin-KK',
                             xy=[np.min(np.log10(self.df[0].f)), np.max(self.KK_rr_im_max)*100*.9],
                             color='k', fontweight='bold')

            ### Figure specifics
            if legend == 'on' or legend == 'potential':
                ax.legend(loc='best',  frameon=False)
            ax.set_xlabel("Z' [$\Omega$]")
            ax.set_ylabel("-Z'' [$\Omega$]")
            if nyq_xlim != 'none':
                ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
            if nyq_ylim != 'none':
                ax.set_ylim(nyq_ylim[0], nyq_ylim[1])
            #Save Figure
            if savefig != 'none':
                fig.savefig(savefig)

        ### Illustrating residuals only

        elif plot == 'residuals':
            colors_real = sns.color_palette("Blues", n_colors=9)
            colors_imag = sns.color_palette("Oranges", n_colors=9)

            ### 1 Cycle
            if len(self.df) == 1:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax = fig.add_subplot(231)
                ax.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3],
                        marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3],
                        marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax.set_xlabel("log(f) [Hz]")
                ax.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]")
                if legend == 'on' or legend == 'potential':
                    ax.legend(loc='best',  frameon=False)
                ax.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and write 'KK-Test' on RR subplot
                self.KK_rr_im_min = np.min(self.KK_rr_im)
                self.KK_rr_im_max = np.max(self.KK_rr_im)
                self.KK_rr_re_min = np.min(self.KK_rr_re)
                self.KK_rr_re_max = np.max(self.KK_rr_re)
                if self.KK_rr_re_max > self.KK_rr_im_max:
                    self.KK_ymax = self.KK_rr_re_max
                else:
                    self.KK_ymax = self.KK_rr_im_max
                if self.KK_rr_re_min < self.KK_rr_im_min:
                    self.KK_ymin = self.KK_rr_re_min
                else:
                    self.KK_ymin = self.KK_rr_im_min
                if np.abs(self.KK_ymin) > self.KK_ymax:
                    ax.set_ylim(self.KK_ymin*100*1.5, np.abs(self.KK_ymin)*100*1.5)
                    if legend == 'on':
                        ax.annotate('Lin-KK, #1',
                                    xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin)*100*1.3],
                                    color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)',
                                    xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin)*100*1.3],
                                    color='k', fontweight='bold')
                elif np.abs(self.KK_ymin) < self.KK_ymax:
                    ax.set_ylim(np.negative(self.KK_ymax)*100*1.5, np.abs(self.KK_ymax)*100*1.5)
                    if legend == 'on':
                        ax.annotate('Lin-KK, #1',
                                    xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax*100*1.3],
                                    color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)',
                                    xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax*100*1.3],
                                    color='k', fontweight='bold')

                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 2 Cycles
            elif len(self.df) == 2:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                #cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])

                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 3 Cycles
            elif len(self.df) == 3:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_xlabel("log(f) [Hz]")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.3], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.3], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.3], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 4 Cycles
            elif len(self.df) == 4:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax2.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                ax3.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')

                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 5 Cycles
            elif len(self.df) == 5:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", )
                ax4.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best',  frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 6 Cycles
            elif len(self.df) == 6:
                fig = figure(figsize=(12, 3.8), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_xlabel("log(f) [Hz]")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best',  frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best',  frameon=False)
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 7 Cycles
            elif len(self.df) == 7:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax3.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax5.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best',  frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best',  frameon=False)
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_xlabel("log(f) [Hz]")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on' or legend == 'potential':
                    ax7.legend(loc='best',  frameon=False)
                ax7.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 8 Cycles
            elif len(self.df) == 8:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)
                ax8 = fig.add_subplot(338)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)
                if legend == 'on' or legend == 'potential':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)
                if legend == 'on' or legend == 'potential':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on' or legend == 'potential':
                    ax5.legend(loc='best',  frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax6.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax6.legend(loc='best',  frameon=False)
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_xlabel("log(f) [Hz]")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=14)
                if legend == 'on' or legend == 'potential':
                    ax7.legend(loc='best',  frameon=False)
                ax7.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 8
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_re[7]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_im[7]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax8.set_xlabel("log(f) [Hz]")
                if legend == 'on' or legend == 'potential':
                    ax8.legend(loc='best',  frameon=False)
                ax8.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[7]) > self.KK_ymax[7]:
                    ax8.set_ylim(self.KK_ymin[7]*100*1.5, np.abs(self.KK_ymin[7])*100*1.5)
                    if legend == 'on':
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[7]) < self.KK_ymax[7]:
                    ax8.set_ylim(np.negative(self.KK_ymax[7])*100*1.5, np.abs(self.KK_ymax[7])*100*1.5)
                    if legend == 'on':
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymax[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK, ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), self.KK_ymax[7]*100*1.2], color='k', fontweight='bold')
                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)

            ### 9 Cycles
            elif len(self.df) == 9:
                fig = figure(figsize=(12, 5), dpi=120, facecolor='w', edgecolor='k')
                fig.subplots_adjust(left=0.1, right=0.95, hspace=0.25, wspace=0.25, bottom=0.1, top=0.95)
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                ax7 = fig.add_subplot(337)
                ax8 = fig.add_subplot(338)
                ax9 = fig.add_subplot(339)

                #cycle 1
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_re[0]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax1.plot(np.log10(self.df[0].f), self.KK_rr_im[0]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax1.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on':
                    ax1.legend(loc='best',  frameon=False)
                ax1.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 2
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_re[1]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax2.plot(np.log10(self.df[1].f), self.KK_rr_im[1]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on':
                    ax2.legend(loc='best',  frameon=False)
                ax2.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 3
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_re[2]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax3.plot(np.log10(self.df[2].f), self.KK_rr_im[2]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on':
                    ax3.legend(loc='best',  frameon=False)
                ax3.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 4
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_re[3]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax4.plot(np.log10(self.df[3].f), self.KK_rr_im[3]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax4.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                if legend == 'on':
                    ax4.legend(loc='best',  frameon=False)
                ax4.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 5
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_re[4]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax5.plot(np.log10(self.df[4].f), self.KK_rr_im[4]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on':
                    ax5.legend(loc='best',  frameon=False)
                ax5.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 6
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_re[5]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax6.plot(np.log10(self.df[5].f), self.KK_rr_im[5]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                if legend == 'on':
                    ax6.legend(loc='best',  frameon=False)
                ax6.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 7
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_re[6]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax7.plot(np.log10(self.df[6].f), self.KK_rr_im[6]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax7.set_ylabel("$\Delta$Z', $\Delta$-Z'' [%]", fontsize=15)
                ax7.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax7.legend(loc='best',  frameon=False)
                ax7.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 8
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_re[7]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax8.plot(np.log10(self.df[7].f), self.KK_rr_im[7]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax8.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax8.legend(loc='best',  frameon=False)
                ax8.axhline(0, ls='--', c='k', alpha=.5)

                # Cycle 9
                ax9.plot(np.log10(self.df[8].f), self.KK_rr_re[8]*100, color=colors_real[3], marker='D', ls='--', ms=6, alpha=.7, label="$\Delta$Z'")
                ax9.plot(np.log10(self.df[8].f), self.KK_rr_im[8]*100, color=colors_imag[3], marker='s', ls='--', ms=6, alpha=.7, label="$\Delta$-Z''")
                ax9.set_xlabel("log(f) [Hz]")
                if legend == 'on':
                    ax9.legend(loc='best',  frameon=False)
                ax9.axhline(0, ls='--', c='k', alpha=.5)

                ### Setting ylims and labeling plot with 'KK-Test' in RR subplot
                self.KK_rr_im_min = []
                self.KK_rr_im_max = []
                self.KK_rr_re_min = []
                self.KK_rr_re_max = []
                self.KK_ymin = []
                self.KK_ymax = []
                for i in range(len(self.df)):
                    self.KK_rr_im_min.append(np.min(self.KK_rr_im[i]))
                    self.KK_rr_im_max.append(np.max(self.KK_rr_im[i]))
                    self.KK_rr_re_min.append(np.min(self.KK_rr_re[i]))
                    self.KK_rr_re_max.append(np.max(self.KK_rr_re[i]))
                    if self.KK_rr_re_max[i] > self.KK_rr_im_max[i]:
                        self.KK_ymax.append(self.KK_rr_re_max[i])
                    else:
                        self.KK_ymax.append(self.KK_rr_im_max[i])
                    if self.KK_rr_re_min[i] < self.KK_rr_im_min[i]:
                        self.KK_ymin.append(self.KK_rr_re_min[i])
                    else:
                        self.KK_ymin.append(self.KK_rr_im_min[i])
                if np.abs(self.KK_ymin[0]) > self.KK_ymax[0]:
                    ax1.set_ylim(self.KK_ymin[0]*100*1.5, np.abs(self.KK_ymin[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymin[0])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax1.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[0])*100*1.5)
                    if legend == 'on':
                        ax1.annotate('Lin-KK, #1', xy=[np.min(np.log10(self.df[0].f)), np.abs(self.KK_ymax[0])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax1.annotate('Lin-KK, ('+str(np.round(np.average(self.df[0].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[0].f)), self.KK_ymax[0]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[1]) > self.KK_ymax[1]:
                    ax2.set_ylim(self.KK_ymin[1]*100*1.5, np.abs(self.KK_ymin[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.3], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), np.max(np.abs(self.KK_ymin[1]))*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[0]) < self.KK_ymax[0]:
                    ax2.set_ylim(np.negative(self.KK_ymax[1])*100*1.5, np.abs(self.KK_ymax[1])*100*1.5)
                    if legend == 'on':
                        ax2.annotate('Lin-KK, #2', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax2.annotate('Lin-KK ('+str(np.round(np.average(self.df[1].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[1].f)), self.KK_ymax[1]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[2]) > self.KK_ymax[2]:
                    ax3.set_ylim(self.KK_ymin[2]*100*1.5, np.abs(self.KK_ymin[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymin[2])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[2]) < self.KK_ymax[2]:
                    ax3.set_ylim(np.negative(self.KK_ymax[0])*100*1.5, np.abs(self.KK_ymax[2])*100*1.5)
                    if legend == 'on':
                        ax3.annotate('Lin-KK, #3', xy=[np.min(np.log10(self.df[2].f)), np.abs(self.KK_ymax[2])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax3.annotate('Lin-KK, ('+str(np.round(np.average(self.df[2].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[2].f)), self.KK_ymax[2]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[3]) > self.KK_ymax[3]:
                    ax4.set_ylim(self.KK_ymin[3]*100*1.5, np.abs(self.KK_ymin[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymin[3])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[3]) < self.KK_ymax[3]:
                    ax4.set_ylim(np.negative(self.KK_ymax[3])*100*1.5, np.abs(self.KK_ymax[3])*100*1.5)
                    if legend == 'on':
                        ax4.annotate('Lin-KK, #4', xy=[np.min(np.log10(self.df[3].f)), np.abs(self.KK_ymax[3])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax4.annotate('Lin-KK, ('+str(np.round(np.average(self.df[3].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[3].f)), self.KK_ymax[3]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[4]) > self.KK_ymax[4]:
                    ax5.set_ylim(self.KK_ymin[4]*100*1.5, np.abs(self.KK_ymin[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymin[4])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[4]) < self.KK_ymax[4]:
                    ax5.set_ylim(np.negative(self.KK_ymax[4])*100*1.5, np.abs(self.KK_ymax[4])*100*1.5)
                    if legend == 'on':
                        ax5.annotate('Lin-KK, #5', xy=[np.min(np.log10(self.df[4].f)), np.abs(self.KK_ymax[4])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax5.annotate('Lin-KK, ('+str(np.round(np.average(self.df[4].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[4].f)), self.KK_ymax[4]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[5]) > self.KK_ymax[5]:
                    ax6.set_ylim(self.KK_ymin[5]*100*1.5, np.abs(self.KK_ymin[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymin[5])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[5]) < self.KK_ymax[5]:
                    ax6.set_ylim(np.negative(self.KK_ymax[5])*100*1.5, np.abs(self.KK_ymax[5])*100*1.5)
                    if legend == 'on':
                        ax6.annotate('Lin-KK, #6', xy=[np.min(np.log10(self.df[5].f)), np.abs(self.KK_ymax[5])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax6.annotate('Lin-KK, ('+str(np.round(np.average(self.df[5].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[5].f)), self.KK_ymax[5]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[6]) > self.KK_ymax[6]:
                    ax7.set_ylim(self.KK_ymin[6]*100*1.5, np.abs(self.KK_ymin[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymin[6])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[6]) < self.KK_ymax[6]:
                    ax7.set_ylim(np.negative(self.KK_ymax[6])*100*1.5, np.abs(self.KK_ymax[6])*100*1.5)
                    if legend == 'on':
                        ax7.annotate('Lin-KK, #7', xy=[np.min(np.log10(self.df[6].f)), np.abs(self.KK_ymax[6])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax7.annotate('Lin-KK, ('+str(np.round(np.average(self.df[6].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[6].f)), self.KK_ymax[6]*100*1.2], color='k', fontweight='bold')
                if np.abs(self.KK_ymin[7]) > self.KK_ymax[7]:
                    ax8.set_ylim(self.KK_ymin[7]*100*1.5, np.abs(self.KK_ymin[7])*100*1.5)
                    if legend == 'on':
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymin[7])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[7]) < self.KK_ymax[7]:
                    ax8.set_ylim(np.negative(self.KK_ymax[7])*100*1.5, np.abs(self.KK_ymax[7])*100*1.5)
                    if legend == 'on':
                        ax8.annotate('Lin-KK, #8', xy=[np.min(np.log10(self.df[7].f)), np.abs(self.KK_ymax[7])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax8.annotate('Lin-KK, ('+str(np.round(np.average(self.df[7].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[7].f)), self.KK_ymax[7]*100*1.2], color='k', fontweight='bold')

                if np.abs(self.KK_ymin[8]) > self.KK_ymax[8]:
                    ax9.set_ylim(self.KK_ymin[8]*100*1.5, np.abs(self.KK_ymin[8])*100*1.5)
                    if legend == 'on':
                        ax9.annotate('Lin-KK, #9', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymin[8])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax9.annotate('Lin-KK ('+str(np.round(np.average(self.df[8].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymin[8])*100*1.2], color='k', fontweight='bold')
                elif np.abs(self.KK_ymin[8]) < self.KK_ymax[8]:
                    ax9.set_ylim(np.negative(self.KK_ymax[8])*100*1.5, np.abs(self.KK_ymax[8])*100*1.5)
                    if legend == 'on':
                        ax9.annotate('Lin-KK, #9', xy=[np.min(np.log10(self.df[8].f)), np.abs(self.KK_ymax[8])*100*1.2], color='k', fontweight='bold')
                    elif legend == 'potential':
                        ax9.annotate('Lin-KK, ('+str(np.round(np.average(self.df[8].E_avg),2))+' V)', xy=[np.min(np.log10(self.df[8].f)), self.KK_ymax[8]*100*1.2], color='k', fontweight='bold')

                #Save Figure
                if savefig != 'none':
                    fig.savefig(savefig)
            else:
                print('Too many spectras, cannot plot all. Maximum spectras allowed = 9')

    def EIS_fit(self, params, circuit, init_only=False, weight_func='modulus', nan_policy='raise'):
        """
        EIS_fit() fits experimental data to an equivalent circuit model using complex non-linear
        least-squares (CNLS) fitting procedure and allows for batch fitting.

        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

        Inputs
        ------------
        - circuit:
          Choose an equivalent circuits and defined circuit as a string. The available circuits
          are the keys of CIRCUIT_DICT in the circuits file.

        - weight_func
          The weight function to which the CNLS fitting is performed
            - modulus (default)
            - unity
            - proportional

        - nan_policy
        How to handle Nan or missing values in dataset
            - ‘raise’ = raise a value error (default)
            - ‘propagate’ = do nothing
            - ‘omit’ = drops missing data

        Returns
        ------------
        Returns the fitted impedance spectra(s) but also the fitted parameters that were used in
        the initial guesses. To call these use e.g. self.fit_Rs
        """
        self.fit = []
        self.circuit_fit = []
        self.init_fit = []
        self.fit_reports = []
        if init_only:
            for param_name in params:
                params[param_name].vary = False
        for i in range(len(self.df)):
            self.fit.append(minimize(leastsq_errorfunc, params, method='leastsq',
                                     args=(self.df[i].w.values,
                                           self.df[i].re.values,
                                           self.df[i].im.values,
                                           circuit,
                                           weight_func), nan_policy=nan_policy, max_nfev=99999))
            self.fit_reports.append(fit_report(self.fit[i]))
            if not init_only:
                final_params = {p: self.fit[i].params[p].value for p in self.fit[i].params}
                self.circuit_fit.append(CIRCUIT_DICT[circuit](params=final_params, w=self.df[i].w))
            init_params = {p: self.fit[i].params[p].init_value for p in self.fit[i].params}
            self.init_fit.append(CIRCUIT_DICT[circuit](params=init_params, w=self.df[i].w))

    def EIS_plot(self, fitting=False, legend=False):
        """
        Plots Experimental and fitted impedance data in three subplots:
            a) Nyquist, b) Bode, c) relative residuals between experimental and fit

        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

        Optional Inputs
        -----------------
        - legend:
            whether or not to show the legend

        - fitting:
          If EIS_fit() has been called. To plot experimental- and fitted data turn fitting on.
          Assumes only one data set!
        """
        init_only = len(self.circuit_fit) == 0
        if fitting and not init_only:
            plot_width = 450
            plot_height = 360
        else:
            plot_width = 620
            plot_height = 500

        # Colors
        if len(self.df) == 1:
            colors = small_palettes['Category20'][10]
            light_colors = (colors[6],)
            dark_colors = (colors[0],)
            fit_colors = [colors[6], colors[1], colors[7]]
            bode_color_left = dark_colors[0]
            bode_color_right = light_colors[0]
        else:
            colors = small_palettes['Category20'][20]
            light_colors = colors[1::2]
            dark_colors = colors[::2]
            fit_colors = []
            bode_color_left = 'black'
            bode_color_right = colors[14]

        # nyquist + bode figure setup
        plot_n = figure(plot_width=plot_width, plot_height=plot_height,
                        tooltips=[("x", "$x"), ("y", "$y")],
                        tools="pan,box_zoom,crosshair,hover,reset,save",
                        x_axis_label="Re(Z) [Ω]", y_axis_label="-Im(Z) [Ω]")
        plot_n.title.text = 'Nyquist'
        plot_n.title.align = 'center'

        plot_b = figure(plot_width=plot_width, plot_height=plot_height,
                        tooltips=[("x", "$x"), ("y", "$y")],
                        tools="pan,box_zoom,crosshair,hover,reset,save",
                        x_axis_label="log(f) [Hz]", y_axis_label="log(|Z|) [Ω]")
        plot_b.title.text = 'Bode'
        plot_b.title.align = 'center'

        max_phase = np.max([np.max(df.Z_phase) for df in self.df])
        min_phase = np.min([np.min(df.Z_phase) for df in self.df])
        max_mag = np.max([np.max(np.log10(df.Z_mag)) for df in self.df])
        min_mag = np.min([np.min(np.log10(df.Z_mag)) for df in self.df])

        plot_b.y_range = Range1d(start=min_mag - 0.01, end=max_mag + 0.01)
        plot_b.extra_y_ranges = {'y2': Range1d(start=min_phase - 1, end=max_phase + 1)}
        plot_b.add_layout(LinearAxis(y_range_name="y2", axis_label="Z phase [deg]"), 'right')

        plot_b.yaxis[0].axis_line_color = bode_color_left
        plot_b.yaxis[0].major_label_text_color = bode_color_left
        plot_b.yaxis[0].major_tick_line_color = bode_color_left
        plot_b.yaxis[0].minor_tick_line_color = bode_color_left
        plot_b.yaxis[0].axis_label_text_color = bode_color_left

        plot_b.yaxis[1].axis_line_color = bode_color_right
        plot_b.yaxis[1].major_label_text_color = bode_color_right
        plot_b.yaxis[1].major_tick_line_color = bode_color_right
        plot_b.yaxis[1].minor_tick_line_color = bode_color_right
        plot_b.yaxis[1].axis_label_text_color = bode_color_right

        if fitting and not init_only:
            # nyquist
            plot_n_resid = figure(plot_width=plot_width, plot_height=100,
                                  x_range=plot_n.x_range,  x_axis_label="Re(Z) [Ω]")
            plot_n.xaxis.visible = False
            plot_n.min_border_bottom = 0
            plot_n_resid.ray(x=[np.mean(self.df[0].re)], y=[0], length=0, angle=0, color='black')
            plot_n_resid.ray(x=[np.mean(self.df[0].re)], y=[0], length=0, angle=np.pi, color='black')

            # bode
            plot_b_resid = figure(plot_width=plot_width, plot_height=100,
                                  x_range=plot_b.x_range, x_axis_label="log(f) [Hz]")
            plot_b.xaxis.visible = False
            plot_b.min_border_bottom = 0
            plot_b_resid.ray(x=[np.mean(np.log10(self.df[0].f))], y=[0], length=0, angle=0, color='black')
            plot_b_resid.ray(x=[np.mean(np.log10(self.df[0].f))], y=[0], length=0, angle=np.pi, color='black')

            magnitude_0 = np.abs(self.circuit_fit[0].values)
            phase_0 = np.angle(self.circuit_fit[0].values, deg=True)
            resid_mag_0 = np.log10(magnitude_0) - np.log10(self.df[0].Z_mag)
            resid_phase_0 = phase_0 - self.df[0].Z_phase

            plot_b_resid.y_range = Range1d(start=np.min(resid_mag_0) - 0.002,
                                           end=np.max(resid_mag_0) + 0.002)
            plot_b_resid.extra_y_ranges = {'y2': Range1d(start=np.min(resid_phase_0) - .15,
                                                         end=np.max(resid_phase_0) + .15)}
            plot_b_resid.add_layout(LinearAxis(y_range_name="y2"), 'right')

            plot_b_resid.yaxis[0].axis_line_color = bode_color_left
            plot_b_resid.yaxis[0].major_label_text_color = bode_color_left
            plot_b_resid.yaxis[0].major_tick_line_color = bode_color_left
            plot_b_resid.yaxis[0].minor_tick_line_color = bode_color_left
            plot_b_resid.yaxis[0].axis_label_text_color = bode_color_left

            plot_b_resid.yaxis[1].axis_line_color = bode_color_right
            plot_b_resid.yaxis[1].major_label_text_color = bode_color_right
            plot_b_resid.yaxis[1].major_tick_line_color = bode_color_right
            plot_b_resid.yaxis[1].minor_tick_line_color = bode_color_right
            plot_b_resid.yaxis[1].axis_label_text_color = bode_color_right

        for i in range(len(self.df)):
            plot_n.scatter(self.df[i].re, self.df[i].im, color=dark_colors[i], legend_label="Data")

            plot_b.scatter(np.log10(self.df[i].f), np.log10(self.df[i].Z_mag),
                           color=dark_colors[i], legend_label="Data ")
            plot_b.scatter(np.log10(self.df[i].f), self.df[i].Z_phase,
                           y_range_name='y2', color=light_colors[i], legend_label="Data")

            if fitting:
                if not init_only:
                    # fit
                    plot_n.line(self.circuit_fit[i].values.real, -self.circuit_fit[i].values.imag,
                                     color=fit_colors[0], legend_label='Fit')
                    magnitude = np.abs(self.circuit_fit[i].values)
                    phase = np.angle(self.circuit_fit[i].values, deg=True)
                    plot_b.line(np.log10(self.df[i].f), np.log10(magnitude),
                                color=fit_colors[1], legend_label="Fit ")
                    plot_b.line(np.log10(self.df[i].f), phase, color=fit_colors[2],
                                y_range_name="y2", legend_label='Fit')
                    # residuals
                    plot_n_resid.scatter(self.circuit_fit[i].values.real,
                                         -self.circuit_fit[i].values.imag - self.df[i].im,
                                         color=dark_colors[i])

                    plot_b_resid.scatter(np.log10(self.df[i].f),
                                         np.log10(magnitude) - np.log10(self.df[i].Z_mag),
                                         color=dark_colors[i])
                    plot_b_resid.scatter(np.log10(self.df[i].f), phase - self.df[i].Z_phase,
                                         color=light_colors[i], y_range_name='y2')

                # initial guess
                plot_n.line(self.init_fit[i].values.real, -self.init_fit[i].values.imag,
                            color='black', line_dash='dashed', legend_label="Init")
                magnitude_init = np.abs(self.init_fit[i].values)
                phase_init = np.angle(self.init_fit[i].values, deg=True)
                plot_b.line(np.log10(self.df[i].f), np.log10(magnitude_init),
                            color=fit_colors[1], line_dash='dashed', legend_label="Init ")
                plot_b.line(np.log10(self.df[i].f), phase_init,
                            color=fit_colors[2], y_range_name="y2",
                            line_dash='dashed', legend_label="Init")

        if legend:
            plot_n.legend.location = 'bottom_right'
            plot_n.legend.click_policy = 'hide'
            plot_b.legend.location = 'center_left'
            plot_b.legend.click_policy = 'hide'
        else:
            plot_n.legend.visible = False
            plot_b.legend.visible = False

        if fitting:
            if not init_only:
                plot_n_resid.yaxis[0].ticker.desired_num_ticks = 3
                nyquist = gridplot([[plot_n], [plot_n_resid]], toolbar_location='right')

                plot_b_resid.yaxis[0].ticker.desired_num_ticks = 3
                plot_b_resid.yaxis[1].ticker.desired_num_ticks = 3
                bode = gridplot([[plot_b], [plot_b_resid]], toolbar_location='right')
            else:
                nyquist = plot_n
                bode = plot_b
            report = self.fit_reports[0].split("[[Correlations]]")[0]
            fr = Div(text=f'<pre>{report}</pre>'.replace('\n', "<br />"), width=400)
            l = layout([[nyquist, bode], [fr]], sizing_mode='scale_width')
        else:
            l = layout(plot_n, plot_b)
        return l

    def plot_Cdl_E(self, interface, BET_Area, m_electrode):
        """
        Normalizing Q to C_eff or Cdl using either norm_nonFara_Q_C() or norm_Fara_Q_C()

        Refs:
            - G. J.Brug, A.L.G. vandenEeden, M.Sluyters-Rehbach, and J.H.Sluyters, J.Elec-
            troanal. Chem. Interfacial Electrochem., 176, 275 (1984)
            - B. Hirschorn, ElectrochimicaActa, 55, 6218 (2010)

        Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

        Inputs
        ---------
        interface = faradaic / nonfaradaic
        BET_Area = BET surface area of electrode material [cm]
        m_electrode = mass of electrode [cm2/mg]

        Inputs
        ---------
        C_eff/C_dl = Normalized Double-layer capacitance measured from impedance [uF/cm2]
        (normalized by norm_nonFara_Q_C() or norm_Fara_Q_C())
        """
        fig = figure(dpi=120, facecolor='w', edgecolor='w')
        fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
        ax = fig.add_subplot(111)

        self.Q_norm = []
        self.E = []
        if interface == 'nonfaradaic':
            self.Q_norm = []
            for i in range(len(self.df)):
                self.Q_norm.append(norm_nonFara_Q_C(Rs=self.fit[i].params.get('Rs').value,
                                                    Q=self.fit[i].params.get('Q').value,
                                                    n=self.fit[i].params.get('n').value))
                self.E.append(np.average(self.df[i].E_avg))

        elif interface == 'faradaic':
            self.Q_norm = []
            for j in range(len(self.df)):
                self.Q_norm.append(norm_Fara_Q_C(Rs=self.fit[j].params.get('Rs').value,
                                                 Rct=self.fit[j].params.get('R').value,
                                                 n=self.fit[j].params.get('n').value,
                                                 fs=self.fit[j].params.get('fs').value,
                                                 L=self.fit[j].params.get('L').value))
                self.E.append(np.average(self.df[j].E_avg))

        self.C_norm = (np.array(self.Q_norm)/(m_electrode*BET_Area))*10**6 #'uF/cm2'
        ax.plot(self.E, self.C_norm, 'o--', label='C$_{dl}$')
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('C$_{dl}$ [$\mu$F/cm$^2$]')


class EIS_sim:
    """
    Simulates and plots Electrochemical Impedance Spectroscopy based-on build-in equivalent cirucit models

    Kristian B. Knudsen (kknu@berkeley.edu || kristianbknudsen@gmail.com)

    Implemented circuits can be found in CIRCUIT_DICT in the circuits file

    Inputs
    --------
    - nyq_xlim/nyq_xlim:
        x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value

    - bode: Plots following Bode plots
        - 'off'
        - 'on' = re, im vs. log(freq)
        - 'log' = log(re, im) vs. log(freq)

        - 're' = re vs. log(freq)
        - 'log_re' = log(re) vs. log(freq)

        - 'im' = im vs. log(freq)
        - 'log_im' = log(im) vs. log(freq)
    """
    def __init__(self, circuit, frange, bode='off', nyq_xlim='none', nyq_ylim='none', legend='on', savefig='none'):
        self.f = frange
        self.w = 2*np.pi*frange
        self.re = circuit.real
        self.im = -circuit.imag

        if bode == 'off':
            fig = figure(dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(111, aspect='equal')

        elif bode in ['on', 'log', 're', 'log_re', 'im', 'log_im', 'log']:
            fig = figure(figsize=(6, 4.5), dpi=120, facecolor='w', edgecolor='w')
            fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
            ax = fig.add_subplot(211, aspect='equal')
            ax1 = fig.add_subplot(212)

        colors = sns.color_palette("colorblind", n_colors=1)
        colors_real = sns.color_palette("Blues", n_colors=1)
        colors_imag = sns.color_palette("Oranges", n_colors=1)

        ### Nyquist Plot
        ax.plot(self.re, self.im, color=colors[0], marker='o', ms=4, lw=2, ls='-', label='Sim')

        ### Bode Plot
        if bode == 'on':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z', -Z'' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 're':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 'log_re':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode=='im':
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("-Z'' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode=='log_im':
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(-Z'') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 'log':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25,  ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25,  ls='-', label="-Z''")
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        ### Figure specifics
        if legend == 'on':
            ax.legend(loc='best',  frameon=False)
        ax.set_xlabel("Z' [$\Omega$]")
        ax.set_ylabel("-Z'' [$\Omega$]")
        if nyq_xlim != 'none':
            ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
        if nyq_ylim != 'none':
            ax.set_ylim(nyq_ylim[0], nyq_ylim[1])

        #Save Figure
        if savefig != 'none':
            fig.savefig(savefig) #saves figure if fix text is given


    def EIS_sim_fit(self,
                    params,
                    circuit,
                    weight_func='modulus',
                    nan_policy='raise',
                    bode='on',
                    nyq_xlim='none',
                    nyq_ylim='none',
                    legend='on',
                    savefig='none'):
        """
        This function fits simulations with a selected circuit. This function is mainly used to
        test fitting functions prior to being used on experimental data

        Kristian B. Knudsen (kknu@berkeley.edu / kristianbknudsen@gmail.com)

        Inputs
        ------------
        - Circuit: Equivlaent circuit models are defined in the CIRCUIT_DICT of the circuits file

        - weight_func = Weight function, Three options:
            - modulus (default)
            - unity
            - proportional

        - nyq_xlim/nyq_xlim: x/y-axis on nyquist plot, if not equal to 'none' state [min,max] value

        - legend: Display legend
            Turn 'on', 'off'

        - bode = Plots Bode Plot - options:
            'on' = re, im vs. log(freq)
            'log' = log(re, im) vs. log(freq)

            're' = re vs. log(freq)
            'log_re' = log(re) vs. log(freq)

            'im' = im vs. log(freq)
            'log_im' = log(im) vs. log(freq)

        Returns
        ------------
        The fitted impedance spectra(s) but also the fitted parameters that were used in the initial
         guesses. To call these use e.g. self.fit_Rs
        """
        self.Fit = minimize(leastsq_errorfunc, params, method='leastsq',
                            args=(self.w, self.re, self.im, circuit, weight_func),
                            max_nfev=99999, nan_policy=nan_policy)
        print(report_fit(self.Fit))

        if circuit in list(CIRCUIT_DICT.keys()):
            self.circuit_fit = CIRCUIT_DICT[circuit](params=self.Fit.params, w=self.w)
        else:
            raise ValueError(f'circuit {circuit} is not a valid option')

        fig = figure(figsize=(6, 4.5), dpi=120, facecolor='w', edgecolor='k')
        fig.subplots_adjust(left=0.1, right=0.95, hspace=0.5, bottom=0.1, top=0.95)
        ax = fig.add_subplot(211, aspect='equal')
        ax1 = fig.add_subplot(212)

        colors = sns.color_palette("colorblind", n_colors=1)
        colors_real = sns.color_palette("Blues", n_colors=1)
        colors_imag = sns.color_palette("Oranges", n_colors=1)

        ### Nyquist Plot
        ax.plot(self.re, self.im, color=colors[0], marker='o', ms=4, lw=2, ls='-', label='Sim')
        ax.plot(self.circuit_fit.real, -self.circuit_fit.imag, lw=0, marker='o', ms=8, mec='r', mew=1, mfc='none', label='Fit')

        ### Bode Plot
        if bode=='on':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), self.circuit_fit.real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.plot(np.log10(self.f), -self.circuit_fit.imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z', -Z'' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 're':
            ax1.plot(np.log10(self.f), self.re, color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z'")
            ax1.plot(np.log10(self.f), self.circuit_fit.real, lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("Z' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 'log_re':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25, ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.circuit_fit.real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode=='im':
            ax1.plot(np.log10(self.f), self.im, color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), -self.circuit_fit.imag, lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("-Z'' [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode=='log_im':
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25, ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), np.log10(-self.circuit_fit.imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(-Z'') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        elif bode == 'log':
            ax1.plot(np.log10(self.f), np.log10(self.re), color=colors_real[0], marker='D', ms=3, lw=2.25,  ls='-', label="Z''")
            ax1.plot(np.log10(self.f), np.log10(self.im), color=colors_imag[0], marker='s', ms=3, lw=2.25,  ls='-', label="-Z''")
            ax1.plot(np.log10(self.f), np.log10(self.circuit_fit.real), lw=0, marker='D', ms=8, mec='r', mew=1, mfc='none', label='Fit')
            ax1.plot(np.log10(self.f), np.log10(-self.circuit_fit.imag), lw=0, marker='s', ms=8, mec='r', mew=1, mfc='none')
            ax1.set_xlabel("log(f) [Hz]")
            ax1.set_ylabel("log(Z', -Z'') [$\Omega$]")
            if legend == 'on':
                ax1.legend(loc='best',  frameon=False)

        ### Figure specifics
        if legend == 'on':
            ax.legend(loc='best',  frameon=False)
        ax.set_xlabel("Z' [$\Omega$]")
        ax.set_ylabel("-Z'' [$\Omega$]")

        if nyq_xlim != 'none':
            ax.set_xlim(nyq_xlim[0], nyq_xlim[1])
        if nyq_ylim != 'none':
            ax.set_ylim(nyq_ylim[0], nyq_ylim[1])

        #Save Figure
        if savefig != 'none':
            fig.savefig(savefig) #saves figure if fix text is given
