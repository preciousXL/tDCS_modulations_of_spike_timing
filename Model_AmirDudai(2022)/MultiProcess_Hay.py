import numpy as np
import neuron
from neuron import h
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PolyCollection
import subprocess
import pickle, glob
from scipy.signal import chirp
from scipy.fftpack import fft
from scipy.optimize import leastsq, curve_fit
import re
import scipy.signal
from scipy.signal import find_peaks
import numba
import time
import multiprocessing

'''generate EPSP-like synaptic current'''
def generate_epsp_like_current(taur=0.5, taud=5, imax=0.5, onset=320, dt=0.025, tstop=600):
    iepsp = np.zeros(int(tstop / dt))
    tpeak = taur * taud * np.log(taur / taud) / (taur - taud)
    adjust = 1 / ((1 - myexp(-tpeak / taur)) - (1 - myexp(-tpeak / taud)))
    amp = adjust * imax
    for i, t in enumerate(np.arange(0, tstop, dt)):
        if t < onset:
            iepsp[i] = 0.
        else:
            a0 = 1 - myexp(-(t - onset) / taur)
            a1 = 1 - myexp(-(t - onset) / taud)
            iepsp[i] = -amp * (a0 - a1)
    return iepsp

def myexp(x):
    if x < -100:
        return 0
    else:
        return np.exp(x)


@numba.njit
def generatePulseCurrent(amp_pulse=0., tstart=0., tend=0., tvar=np.arange(0, 1, 0.01)):
    index_pulse = np.logical_and(tvar >= tstart, tvar <= tend)
    Ipulse = np.zeros_like(tvar)
    Ipulse[index_pulse] = amp_pulse
    return Ipulse


@numba.njit
def isCalciumSpike(tvar, vdend):
    CaSpike_startVth = -40
    CaSpike_endVth = -42
    logical_above_startVth = vdend > CaSpike_startVth
    logical_above_endVth = vdend > CaSpike_endVth
    diff_logical_above_startVth = np.diff(logical_above_startVth)
    diff_logical_above_endVth = np.diff(logical_above_endVth)
    bool, CaSpike_start_t, CaSpike_end_t = 0, None, None
    if np.any(diff_logical_above_startVth):
        bool = 1
        CaSpike_start_t = tvar[np.where(diff_logical_above_startVth)][0]
        CaSpike_end_t = tvar[:-1][np.where(diff_logical_above_endVth)][-1]
    return bool, CaSpike_start_t, CaSpike_end_t

@numba.njit
def calcICaTime(tvar, ICa, ICaVth=-0.03):
    index_below_ICaVth = ICa < ICaVth
    diff_index_below_ICaVth = np.diff(index_below_ICaVth)
    bool, ICa_start_t, ICa_end_t = 0, None, None
    if np.any(diff_index_below_ICaVth):
        bool        = 1
        ICa_start_t = tvar[np.where(diff_index_below_ICaVth)][0]
        ICa_end_t   = tvar[np.where(diff_index_below_ICaVth)][1]
    return bool, ICa_start_t, ICa_end_t

def calcSpikeNumberAndSpikeTime(tvar, vsoma):
    risingBefore = np.hstack((0, vsoma[1:] - vsoma[:-1])) > 0  # v(t)-v(t-1)>0
    fallingAfter = np.hstack((vsoma[1:] - vsoma[:-1], 0)) < 0  # v(t)-v(t+1)<0
    localMaximum = np.logical_and(fallingAfter, risingBefore)
    largerThanThresh = vsoma > 0.
    binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)
    spikeInds = np.nonzero(binarySpikeVector)
    spikeNumber = np.sum(binarySpikeVector)
    outputSpikeTimes = tvar[spikeInds]
    return spikeNumber, outputSpikeTimes


class L5PTcell():
    def __init__(self):
        self.init_parameters()
        self.create_cell()
        self.pre_calc_for_plotmorph()
        self.calc_each_segment_vinit()

    def init_parameters(self):
        self.diam_scale_factor = 0
        self.diam_plus_factor = 2
        self.subsection_names = ['soma', 'dend', 'apic', 'axon']
        self.subsection_colors = ['k', 'g', 'b', 'r']
        self.somaAtOrigin = False
        self.shiftx, self.shifty, self.shiftz = 0.0, 0.0, 0.0

    def create_cell(self):
        h.load_file('nrngui.hoc')
        h.load_file("import3d.hoc")
        morphologyFilename = "morphologies/cell1.asc"
        biophysicalModelFilename = "L5PCbiophys5b.hoc"
        biophysicalModelTemplateFilename = "L5PCtemplate_2.hoc"
        EfieldFilename = "getes.hoc"

        h.load_file(biophysicalModelFilename)
        h.load_file(biophysicalModelTemplateFilename)
        self.cell = h.L5PCtemplate(morphologyFilename)
        h.load_file("steadystate_init.hoc")
        h.load_file(EfieldFilename)

    def create_SectionLists(self):
        '''Create section lists for different kinds of sections'''
        self.allsecnames = []
        self.allseclists = h.SectionList()
        for sec in self.cell.all:
            self.allseclists.append(sec=sec)
            self.allsecnames.append(sec.name())
        self.totnsecs = len(self.allsecnames)
        # self.calc_totnsegs()

    def calc_totnsegs(self):
        '''calculate the segment number of all section'''
        self.totnsegs = 0
        for sec in self.allseclists:
            self.totnsegs += sec.nseg
        '''calculate the segment number of each subsection, like soma'''
        # ['soma',  'axon', 'dend', 'apic']
        self.subsection_nsegs = []
        self.each_subsection = []
        self.sec_nseg = []
        num, idx, temp = 0, 0, 0
        num2 = 0
        for sec in self.allseclists:
            self.sec_nseg.append(sec.nseg)
            temp += sec.nseg
            if self.subsection_names[idx] in str(sec):
                num += sec.nseg
                num2 += 1
            else:
                self.subsection_nsegs.append(num)
                self.each_subsection.append(num2)
                num = sec.nseg
                num2 = 1
                idx += 1
            if temp == self.totnsegs:
                self.subsection_nsegs.append(num)
                self.each_subsection.append(num2)
        self.subsection_nsegs_end = []
        for i in range(len(self.subsection_nsegs) + 1):
            self.subsection_nsegs_end.append(sum(self.subsection_nsegs[:i]))

    def calc_eachsection_nsegs(self):
        if not hasattr(self, 'allseclists'):
            self.create_SectionLists()
        '''计算每个section中segment的索引，shape=[nsec, 2], 2代表该section开始和结尾的segment索引'''
        self.each_section_index = np.zeros((len(self.allsecnames), 2))
        secidx, numseg = 0, 0
        for sec in self.allseclists:
            self.each_section_index[secidx, 0] = numseg
            numseg += sec.nseg
            self.each_section_index[secidx, 1] = numseg
            secidx += 1
        self.each_section_index = self.each_section_index.astype(np.int64)

    def collect_geometry(self):
        if not hasattr(self, 'x'):
            # 如果没有定义self.x，则声明变量
            self.x = None
            self.y = None
            self.z = None
            self.area = None
            self.d = None
            self.length = None
        self.collect_geometry_neuron()

    def collect_geometry_neuron(self):
        '''Loop over allseclist to determine area, diam, xyz-start- and endpoints, embed geometry to cell object'''
        areavec = np.zeros(self.totnsegs)
        diamvec = np.zeros(self.totnsegs)
        lengthvec = np.zeros(self.totnsegs)
        xstartvec = np.zeros(self.totnsegs)
        xendvec = np.zeros(self.totnsegs)
        ystartvec = np.zeros(self.totnsegs)
        yendvec = np.zeros(self.totnsegs)
        zstartvec = np.zeros(self.totnsegs)
        zendvec = np.zeros(self.totnsegs)

        counter = 0
        # loop over all segments
        for sec in self.allseclists:
            n3d = int(neuron.h.n3d(sec=sec))  # or int(sec.n3d())
            nseg = sec.nseg  # sec.n3d() > sec.nseg
            gsen2 = 1. / 2 / nseg
            if n3d > 0:
                # create interpolation objects for the xyz pt3d info:
                L, x, y, z = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d), np.zeros(n3d)
                for i in range(n3d):
                    L[i] = neuron.h.arc3d(i, sec=sec)  # return the length of the first i-th n3d()
                    x[i] = neuron.h.x3d(i, sec=sec)
                    y[i] = neuron.h.y3d(i, sec=sec)
                    z[i] = neuron.h.z3d(i, sec=sec)
                # normalize as seg.x [0, 1]
                L /= sec.L
                # temporary store position of segment midpoints
                segx = np.zeros(nseg)
                for i, seg in enumerate(sec):
                    segx[i] = seg.x
                # can't be >0 which may happen due to NEURON->Python float
                # transfer:
                segx0 = (segx - gsen2).round(decimals=6)
                segx1 = (segx + gsen2).round(decimals=6)
                # fill vectors with interpolated coordinates of start and end
                # points
                xstartvec[counter:counter + nseg] = np.interp(segx0, L, x)
                xendvec[counter:counter + nseg] = np.interp(segx1, L, x)
                ystartvec[counter:counter + nseg] = np.interp(segx0, L, y)
                yendvec[counter:counter + nseg] = np.interp(segx1, L, y)
                zstartvec[counter:counter + nseg] = np.interp(segx0, L, z)
                zendvec[counter:counter + nseg] = np.interp(segx1, L, z)
                # fill in values area, diam, length
                for seg in sec:
                    areavec[counter] = neuron.h.area(seg.x, sec=sec)
                    diamvec[counter] = seg.diam
                    lengthvec[counter] = sec.L / nseg
                    counter += 1
        # set cell attributes
        self.x = np.c_[xstartvec, xendvec]
        self.y = np.c_[ystartvec, yendvec]
        self.z = np.c_[zstartvec, zendvec]
        self.area = areavec
        self.d = diamvec * self.diam_scale_factor + self.diam_plus_factor
        self.length = lengthvec

    def get_idx_polygons(self, projection=('x', 'z')):
        '''For each segment idx in cell create a polygon in the plane determined by the projection kwarg (default ('x', 'z'))'''
        ''' example:
        from matplotlib.collections import PolyCollection
        zips = []
        for x, z in self.get_idx_polygons(projection=('x', 'z')):
            zips.append(list(zip(x, z)))
        polycol = PolyCollection(zips, edgecolors='none', facecolors='gray')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_collection(polycol)
        ax.axis(ax.axis('equal'))
        '''
        if len(projection) != 2:
            raise ValueError("projection arg be a tuple like ('x', 'y')")
        if 'x' in projection and 'y' in projection:
            pass
        elif 'x' in projection and 'z' in projection:
            pass
        elif 'y' in projection and 'z' in projection:
            pass
        else:
            mssg = "projection must be a length 2 tuple of 'x', 'y' or 'z'!"
            raise ValueError(mssg)

        polygons = []
        for i in np.arange(self.totnsegs):
            polygons.append(self.create_segment_polygon(i, projection))

        return polygons

    def create_segment_polygon(self, i, projection=('x', 'z')):
        '''Create a polygon to fill for segment i, in the plane determined by kwarg projection'''
        x = getattr(self, projection[0])[i]  # 等同于x = self.x， x[i].shape = (1, 2)
        z = getattr(self, projection[1])[i]
        d = self.d[i]
        # calculate angles
        dx = np.diff(x)
        dz = np.diff(z)
        theta = np.arctan2(dz, dx)[0]
        x = np.r_[x, x[::-1]]  # x=(xstart, xend), x[::-1]=(xend, xstart) →→→ x.shape=(1, 4)
        z = np.r_[z, z[::-1]]
        # 1st corner:
        x[0] -= 0.5 * d * np.sin(theta)
        z[0] += 0.5 * d * np.cos(theta)
        # end of section, first side
        x[1] -= 0.5 * d * np.sin(theta)
        z[1] += 0.5 * d * np.cos(theta)
        # other side
        # end of section, second side
        x[2] += 0.5 * d * np.sin(theta)
        z[2] -= 0.5 * d * np.cos(theta)
        # last corner:
        x[3] += 0.5 * d * np.sin(theta)
        z[3] -= 0.5 * d * np.cos(theta)
        return x, z  # x.shape=y.shape=(1, 4) 矩形四个顶点坐标，顺时针方向

    def make_soma_at_origin(self, somaAtOrigin=True):
        if somaAtOrigin == True:
            self.shiftx = np.mean(self.x[0])
            self.shifty = np.mean(self.y[0])
            self.shiftz = np.mean(self.z[0])
            self.x = self.x - self.shiftx
            self.y = self.y - self.shifty
            self.z = self.z - self.shiftz
        elif somaAtOrigin == False:
            self.shiftx, self.shifty, self.shiftz = 0.0, 0.0, 0.0

    def get_section_coordinate(self, sec, xin, projection=('x', 'y')):
        n3d = int(neuron.h.n3d(sec=sec))
        nseg = sec.nseg
        gsen2 = 1. / 2 / nseg
        if n3d > 0:
            L, x, y, z = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d), np.zeros(n3d)
            for i in range(n3d):
                L[i] = neuron.h.arc3d(i, sec=sec)
                x[i] = neuron.h.x3d(i, sec=sec)
                y[i] = neuron.h.y3d(i, sec=sec)
                z[i] = neuron.h.z3d(i, sec=sec)
            L /= sec.L
            sec_x = np.interp(xin, L, x) - self.shiftx
            sec_y = np.interp(xin, L, y) - self.shifty
            sec_z = np.interp(xin, L, z) - self.shiftz
            coordinate = {'x': sec_x, 'y': sec_y, 'z': sec_z}
        return np.array([coordinate[projection[0]], coordinate[projection[1]]])

    def pre_calc_for_plotmorph(self, projection=('x', 'y')):
        if not hasattr(self, 'allsecnames'):
            self.create_SectionLists()
        self.calc_totnsegs()
        self.collect_geometry()
        self.make_soma_at_origin(somaAtOrigin=self.somaAtOrigin)
        self.calc_eachsection_nsegs()
        self.morph_zips = []
        for x, y in self.get_idx_polygons(projection=projection):
            self.morph_zips.append(list(zip(x, y)))

    def plot_neuron_morphology(self, ax, morph_zips, ec='none', fc='k'):
        from matplotlib.collections import PolyCollection
        polycol = PolyCollection(list(morph_zips), edgecolors=ec, facecolors=fc)
        ax.add_collection(polycol)

    def create_recordings_allseg(self, dt=0.025):
        if not hasattr(self, 'totnsegs'):
            self.totnsegs = 0
            for sec in self.cell.all:
                self.totnsegs += sec.nseg
        self.allseg_recordings = {}
        num = 0
        for sec in self.cell.all:
            for seg in sec:
                self.allseg_recordings['voltage_segment_%d' % num] = neuron.h.Vector().record(sec(seg.x)._ref_v, dt)
                num += 1

    def get_all_segment_voltage(self):
        '''return the membrane voltage of each segment'''
        len_t = len(self.allseg_recordings[list(self.allseg_recordings.keys())[0]].as_numpy().copy())
        voltage_segment = np.zeros((self.totnsegs, len_t))
        list_keys = list(self.allseg_recordings.keys())
        for i in range(len(list_keys)):
            voltage_segment[i, :] = self.allseg_recordings[list_keys[i]].as_numpy().copy()
        return voltage_segment

    def get_segment_init_voltage(self):
        if not hasattr(self, 'allseg_resting_voltage'):
            dt = 0.025
            h.celsius = 34
            h.dt = dt
            h.tstop = 2e3
            neuron.h.stdinit()
            neuron.h.cvode_active(0)
            neuron.h.run()
            segvoltage = self.get_all_segment_voltage()
            self.allseg_resting_voltage = segvoltage[:, -1]
            np.save('data/allseg_resting_voltage.npy', self.allseg_resting_voltage)

    def calc_each_segment_vinit(self):
        if not os.path.exists('data/allseg_resting_voltage.npy'):
            self.create_recordings_allseg()
            self.get_segment_init_voltage()
        self.allseg_resting_voltage = np.load('data/allseg_resting_voltage.npy')  # shape=(706,)
        self.segment_init_vm = []
        idx = 0
        for sec in h.allsec():
            nseg = sec.nseg
            temp = self.allseg_resting_voltage[idx:idx + nseg]
            self.segment_init_vm += [temp[0]] + temp.tolist() + [temp[-1]]
            idx += nseg
        self.segment_init_vm = np.array(self.segment_init_vm)

    def add_recordings(self, secname='apic', secidx=0, x=0.5):
        if not hasattr(self, 'recordings'):
            self.recordings = {}
        if 't' not in list(self.recordings.keys()):
            self.recordings['t'] = h.Vector().record(h._ref_t)
        tempname = secname + '[' + str(secidx) + ']' + '(' + str(x) + ')'
        if secname == 'soma':
            self.recordings[tempname] = h.Vector().record(self.cell.soma[secidx](x)._ref_v)
        elif secname == 'apic':
            self.recordings[tempname] = h.Vector().record(self.cell.apic[secidx](x)._ref_v)
        elif secname == 'dend':
            self.recordings[tempname] = h.Vector().record(self.cell.dend[secidx](x)._ref_v)
        elif secname == 'axon':
            self.recordings[tempname] = h.Vector().record(self.cell.axon[secidx](x)._ref_v)

    def run_simulation_EF(self, Evar=0., tvar=0., dt=0.025, duration=0., theta=90, phi=90):
        '''Simulation'''
        h.calcesE(theta, phi)
        h.dt = dt
        h.tstop = duration
        h.setstim_snowp()
        h.stim_amp.from_python(Evar)
        h.stim_time.from_python(tvar)
        h.attach_stim()

        # neuron.h.stdinit()
        # neuron.h.cvode_active(0)
        neuron.h.run()


cell = L5PTcell()

'''添加远端EPSP-like电流刺激'''
Id_Apic36_9 = h.IClamp(cell.cell.apic[36](0.9))
Id_Apic36_9.delay = 0
Id_Apic36_9.dur   = 1e9
Id_Apic36_9.amp   = 0
'''添加胞体刺激'''
Is_Soma0_5 = h.IClamp(cell.cell.soma[0](0.5))
Is_Soma0_5.delay = 0
Is_Soma0_5.dur   = 1e9
Is_Soma0_5.amp   = 0
'''定义记录电压信息'''
cell.add_recordings(secname='soma', secidx=0, x=0.5)
cell.add_recordings(secname='apic', secidx=36, x=0.9)
cell.add_recordings(secname='apic', secidx=36, x=0.5)
cell.add_recordings(secname='apic', secidx=36, x=0.1)
'''定义记录电流和变量信息'''
cell.recordings['Id_Apic36_9'] = h.Vector().record(Id_Apic36_9._ref_i)
cell.recordings['Is_Soma0_5']  = h.Vector().record(Is_Soma0_5._ref_i)
cell.recordings['ICa']         = h.Vector().record(cell.cell.apic[36](0.9).ca_ion._ref_ica)
cell.recordings['ICaHVA']      = h.Vector().record(cell.cell.apic[36](0.9).Ca_HVA._ref_ica)
cell.recordings['ICaLVA']      = h.Vector().record(cell.cell.apic[36](0.9).Ca_LVAst._ref_ica)
cell.recordings['ISK']      = h.Vector().record(cell.cell.apic[36](0.9).SK_E2._ref_ik)
cell.recordings['ISKv3']      = h.Vector().record(cell.cell.apic[36](0.9).SKv3_1._ref_ik)
cell.recordings['Im']      = h.Vector().record(cell.cell.apic[36](0.9).Im._ref_ik)
cell.recordings['Ih']      = h.Vector().record(cell.cell.apic[36](0.9).Ih._ref_ihcn)
cell.recordings['INaTs2_t']      = h.Vector().record(cell.cell.apic[36](0.9).NaTs2_t._ref_ina)
cell.recordings['IK']      = h.Vector().record(cell.cell.apic[36](0.9).k_ion._ref_ik)
cell.recordings['INa']      = h.Vector().record(cell.cell.apic[36](0.9).na_ion._ref_ina)
cell.recordings['ICaHVA_m'] = h.Vector().record(cell.cell.apic[36](0.9).Ca_HVA._ref_m)
cell.recordings['ICaHVA_h'] = h.Vector().record(cell.cell.apic[36](0.9).Ca_HVA._ref_h)

recordingsKeysList = list(cell.recordings.keys())



def calcTdcsOnCaSpike(paraImax, paraVds, paraTde):
    dt = 0.025
    DEL, DUR = 510, 150 - 10
    tstop = DEL + DUR + 0
    tvar = np.arange(0, tstop, dt)
    # 电场参数
    AMP_AC, FREQ = 0., 10
    AMP_DC = paraVds
    Evar_DEL = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR = AMP_DC + AMP_AC * np.sin(2 * np.pi * FREQ * np.arange(0, DUR, dt) / 1000)
    Evar_tstop = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    # 突触刺激
    Iepsp = -generate_epsp_like_current(taur=0.5, taud=5, imax=paraImax, onset=DEL + paraTde, dt=dt, tstop=tstop)
    tvar_hoc = h.Vector().from_python(tvar)
    Iepsp_hoc = h.Vector().from_python(Iepsp)
    Iepsp_hoc.play(Id_Apic36_9._ref_amp, tvar_hoc, True)
    Is_Soma0_5.delay = 500
    Is_Soma0_5.dur = 1e9
    Is_Soma0_5.amp = 0
    # 仿真
    cell.run_simulation_EF(Evar=Evar, tvar=tvar, dt=dt, duration=tstop, theta=90, phi=90)
    # 提取仿真结果
    idx1, idx2 = 25, int(tstop / dt)
    t     = cell.recordings['t'].to_python()[idx1:idx2]
    vsoma = cell.recordings['soma[0](0.5)'].to_python()[idx1:idx2]
    vdend = cell.recordings['apic[36](0.9)'].to_python()[idx1:idx2]
    ICa   = cell.recordings['ICa'].to_python()[idx1:idx2]
    t, vsoma, vdend, ICa = np.array(t), np.array(vsoma), np.array(vdend), np.array(ICa)
    CaPeakAmplitude, CaPeakTime = np.max(vdend), t[np.argmax(vdend)]
    CaSpikeBool, ts, te = isCalciumSpike(t, vdend)
    CaStartTime, CaEndTime, CaDuration = 0, 0, 0
    if CaSpikeBool == 1:
        CaStartTime, CaEndTime, CaDuration = ts, te, te-ts

    spikeNumber, spikeTime = calcSpikeNumberAndSpikeTime(t, vsoma)
    firstSpikeTime, secondSpikeTime, firingRateWithinBurst = 0, 0, 0
    if spikeNumber > 0:
        firstSpikeTime = spikeTime[0]
    if spikeNumber > 1:
        secondSpikeTime = spikeTime[1]
        firingRateWithinBurst = 1e3 / np.mean(np.diff(spikeTime))

    ICaPeakAmplitude, ICaPeakTime = np.min(ICa), t[np.argmin(ICa)]
    ICaBool, ts, te = calcICaTime(t, ICa, ICaVth=-0.03)
    ICaStartTime, ICaEndTime, ICaDuration = 0, 0, 0
    if ICaBool == 1:
        ICaStartTime, ICaEndTime, ICaDuration = ts, te, te-ts


    return paraImax, paraVds, paraTde, \
           CaSpikeBool, CaStartTime, CaEndTime, CaDuration, CaPeakAmplitude, CaPeakTime, \
           spikeNumber, firstSpikeTime, secondSpikeTime, firingRateWithinBurst, \
           ICaStartTime, ICaEndTime, ICaDuration, ICaPeakAmplitude, ICaPeakTime


def calcTdcsOnCaSpike_new(paraImax, paraVds, paraTde, paraEtheta):
    dt = 0.025
    DEL, DUR = 510, 350
    tstop = DEL + DUR + 0
    tvar = np.arange(0, tstop, dt)
    # 电场参数
    AMP_AC, FREQ = 0., 10
    AMP_DC = paraVds
    Evar_DEL = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR = AMP_DC + AMP_AC * np.sin(2 * np.pi * FREQ * np.arange(0, DUR, dt) / 1000)
    Evar_tstop = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))
    # 突触刺激
    Iepsp = -generate_epsp_like_current(taur=0.5, taud=5, imax=paraImax, onset=DEL + paraTde, dt=dt, tstop=tstop)
    tvar_hoc = h.Vector().from_python(tvar)
    Iepsp_hoc = h.Vector().from_python(Iepsp)
    Iepsp_hoc.play(Id_Apic36_9._ref_amp, tvar_hoc, True)
    Is_Soma0_5.delay = 500
    Is_Soma0_5.dur = 1e9
    Is_Soma0_5.amp = 0
    # 仿真
    cell.run_simulation_EF(Evar=Evar, tvar=tvar, dt=dt, duration=tstop, theta=paraEtheta, phi=90)
    # 提取仿真结果
    idx1, idx2 = 25, int(tstop / dt)
    t     = cell.recordings['t'].to_python()[idx1:idx2]
    vsoma = cell.recordings['soma[0](0.5)'].to_python()[idx1:idx2]
    vdend = cell.recordings['apic[36](0.9)'].to_python()[idx1:idx2]
    ICa   = cell.recordings['ICa'].to_python()[idx1:idx2]
    t, vsoma, vdend, ICa = np.array(t), np.array(vsoma), np.array(vdend), np.array(ICa)
    CaPeakAmplitude, CaPeakTime = np.max(vdend), t[np.argmax(vdend)]
    CaSpikeBool, ts, te = isCalciumSpike(t, vdend)
    CaStartTime, CaEndTime, CaDuration = 0, 0, 0
    if CaSpikeBool == 1:
        CaStartTime, CaEndTime, CaDuration = ts, te, te-ts

    spikeNumber, spikeTime = calcSpikeNumberAndSpikeTime(t, vsoma)
    firstSpikeTime, secondSpikeTime, firingRateWithinBurst = 0, 0, 0
    if spikeNumber > 0:
        firstSpikeTime = spikeTime[0]
    if spikeNumber > 1:
        secondSpikeTime = spikeTime[1]
        firingRateWithinBurst = 1e3 / np.mean(np.diff(spikeTime))

    ICaPeakAmplitude, ICaPeakTime = np.min(ICa), t[np.argmin(ICa)]
    ICaBool, ts, te = calcICaTime(t, ICa, ICaVth=-0.03)
    ICaStartTime, ICaEndTime, ICaDuration = 0, 0, 0
    if ICaBool == 1:
        ICaStartTime, ICaEndTime, ICaDuration = ts, te, te-ts


    return paraImax, paraVds, paraTde, \
           CaSpikeBool, CaStartTime, CaEndTime, CaDuration, CaPeakAmplitude, CaPeakTime, \
           spikeNumber, firstSpikeTime, secondSpikeTime, firingRateWithinBurst, \
           ICaStartTime, ICaEndTime, ICaDuration, ICaPeakAmplitude, ICaPeakTime




if __name__ == '__main__':
    start_time = time.time()
    casemark = 100
    if casemark == 1:
        list_Imax = [1.32]
        list_Vds = np.arange(-15, 15.1, 1)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        np.save('data/Imax1.32nA_tde20ms_varyVds-15-15mV.npy', res)

    if casemark == 2:
        list_Imax = [1.6]
        list_Vds = np.arange(-15, 15.1, 1)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        np.save('data/Imax1.6nA_tde20ms_varyVds-15-15mV.npy', res)

    if casemark == 3:
        list_Imax = np.arange(1, 2.01, 0.02)
        list_Vds = np.arange(-10, 10.1, 0.5)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_Imax), len(list_Vds), len(list_tde))).tolist()
        for i in range(len(list_Imax)):
            for j in range(len(list_Vds)):
                for k in range(len(list_tde)):
                                results[i][j][k] = res[num]
                                num = num + 1
        results = np.array(results)

        np.save('data/Imax1-2nA_tde20ms_varyVds-10-10mV.npy', results)

    if casemark == 4:
        list_Imax = [1.32, 1.6]
        list_Vds = [0, 2.5, 5]
        list_tde = [0, 20, 300]
        listEfieldDirection = np.arange(-90, 270+1, 10)
        paras = [[i, j, k, m] for i in list_Imax for j in list_Vds for k in list_tde for m in listEfieldDirection]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcTdcsOnCaSpike_new, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_Imax), len(list_Vds), len(list_tde), len(listEfieldDirection))).tolist()
        for i in range(len(list_Imax)):
            for j in range(len(list_Vds)):
                for k in range(len(list_tde)):
                    for m in range(len(listEfieldDirection)):
                        results[i][j][k][m] = res[num]
                        num = num + 1
        results = np.array(results)
        np.save('data/Imax1.32_1.6nA_tde0-20-300ms_Vd0-2.5-5smV_EFDirection-180-180degree.npy', results)

    if casemark == 5:
        list_Imax = [1.32, 1.6]
        list_Vds = [-5, -2.5, 0, 2.5, 5]
        list_tde = np.arange(0, 300+1, 10)
        listEfieldDirection = [90]
        paras = [[i, j, k, m] for i in list_Imax for j in list_Vds for k in list_tde for m in listEfieldDirection]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcTdcsOnCaSpike_new, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_Imax), len(list_Vds), len(list_tde), len(listEfieldDirection))).tolist()
        for i in range(len(list_Imax)):
            for j in range(len(list_Vds)):
                for k in range(len(list_tde)):
                    for m in range(len(listEfieldDirection)):
                        results[i][j][k][m] = res[num]
                        num = num + 1
        results = np.array(results)
        np.save('data/Imax1.32_1.6nA_tde0-300ms_Vds+-5+-2.5mV_EFDirection90degree.npy', results)

    casemark = 6
    if casemark == 6:
        # This code is usde to calculate the spike timing changing as a function of field intensity and current amplitude
        list_Imax = np.arange(1, 2.01, 0.01)
        list_Vds = np.arange(-5, 5.01, 0.2)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=4)
        res = pool.starmap(calcTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_Imax), len(list_Vds), len(list_tde))).tolist()
        for i in range(len(list_Imax)):
            for j in range(len(list_Vds)):
                for k in range(len(list_tde)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)

        np.save('data/Hay_Imax1-2nA_tde20ms_Vds-5-5mV.npy', results)
        print('Running time:', time.time() - start_time)


