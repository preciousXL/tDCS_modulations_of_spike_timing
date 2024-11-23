import time
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
import numba
import scipy.signal
from scipy.signal import find_peaks
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

def generateOUCurrent(miu=0., sigma=0., tvar=np.arange(0, 10, 0.1)):
    tau = 3
    dt = tvar[1] - tvar[0]
    Istim = np.zeros_like(tvar)
    Istim[0] = miu
    for i in range(len(tvar) - 1):
        Istim[i + 1] = Istim[i] + dt * (miu - Istim[i]) / tau + sigma * np.random.randn() * np.sqrt(2 * dt / tau)
    return Istim

def calcSpikeNumberAndSpikeTime(tvar, vsoma):
    risingBefore = np.hstack((0, vsoma[1:] - vsoma[:-1])) > 0  # v(t)-v(t-1)>0
    fallingAfter = np.hstack((vsoma[1:] - vsoma[:-1], 0)) < 0  # v(t)-v(t+1)<0
    localMaximum = np.logical_and(fallingAfter, risingBefore)  # 逻辑与，上述两者逻辑与为真代表为局部最大值，是放电峰值可能存在的时刻
    largerThanThresh = vsoma > 0.  # 定义一个远大于放电阈值的电压值
    binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)  # 放电峰值时刻二进制序列
    spikeInds = np.nonzero(binarySpikeVector)
    spikeNumber = np.sum(binarySpikeVector)
    outputSpikeTimes = tvar[spikeInds]

    return spikeNumber, outputSpikeTimes

@numba.njit
def isCalciumSpike(tvar, vdend, CaSpike_startVth=-30, CaSpike_endVth=-50):
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
def calcICaTime(tvar, ICa, ICaVth=-0.12):
    index_below_ICaVth = ICa < ICaVth
    diff_index_below_ICaVth = np.diff(index_below_ICaVth)
    bool, ICa_start_t, ICa_end_t = 0, None, None
    if np.any(diff_index_below_ICaVth):
        bool = 1
        ICa_start_t = tvar[np.where(diff_index_below_ICaVth)][0]
        ICa_end_t = tvar[np.where(diff_index_below_ICaVth)][1]
    return bool, ICa_start_t, ICa_end_t


class L5PTcell():
    def __init__(self):
        self.init_parameters()
        self.create_cell()
        self.createSectionList()

    def init_parameters(self):
        self.diam_scale_factor = 0
        self.diam_plus_factor = 2
        self.subsection_names = ['soma', 'apic', 'dend', 'axon']
        self.subsection_colors = ['k', 'b', 'g', 'r']
        self.eachSubsectionIndex = []
        self.eachSubsectionIndex.append([0, 1])
        self.eachSubsectionIndex.append([1, 80])
        self.eachSubsectionIndex.append([80, 130])
        self.eachSubsectionIndex.append([130, 142])
        self.somaAtOrigin = False
        self.shiftx, self.shifty, self.shiftz = 0.0, 0.0, 0.0

    def create_cell(self):
        h.load_file('nrngui.hoc')
        h.load_file('init_snowp.hoc')
        h.load_file("steadystate_init.hoc")
        h.load_file("getes.hoc")

    def createSectionList(self):
        self.allSectionList = [sec for sec in h.allsec()]
        self.allSectionNames = [sec.name() for sec in self.allSectionList]
        self.totalSectionNumber = len(self.allSectionList)
        self.somaSectionList = self.allSectionList[self.eachSubsectionIndex[0][0]:self.eachSubsectionIndex[0][1]]
        self.apicalSectionList = self.allSectionList[self.eachSubsectionIndex[1][0]:self.eachSubsectionIndex[1][1]]
        self.basalSectionList = self.allSectionList[self.eachSubsectionIndex[2][0]:self.eachSubsectionIndex[2][1]]
        self.axonalSectionList = self.allSectionList[self.eachSubsectionIndex[3][0]:]
        self.eachSectionSegmentNumber = np.array([sec.nseg for sec in self.allSectionList])
        self.totalSegmentNumber = np.sum(self.eachSectionSegmentNumber)
        self.totnsegs = self.totalSegmentNumber
        self.eachSectionSegmentIndex = []
        idx0, idx1 = 0, 0
        for i in range(self.totalSectionNumber):
            idx1 += self.eachSectionSegmentNumber[i]
            self.eachSectionSegmentIndex.append([idx0, idx1])
            idx0 = idx1

    def pre_calc_for_plotmorph(self, projection=('x', 'y')):
        if not hasattr(self, 'allSectionList'):
            self.createSectionList()
        self.collect_geometry()
        self.make_soma_at_origin(somaAtOrigin=self.somaAtOrigin)
        self.morph_zips = []
        for x, y in self.get_idx_polygons(projection=projection):
            self.morph_zips.append(list(zip(x, y)))

    def collect_geometry(self):
        if not hasattr(self, 'x'):
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
        for sec in self.allSectionList:
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

    def get_idx_polygons(self, projection=('x', 'z')):
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
        x = getattr(self, projection[0])[i]
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
        return x, z  # x.shape=y.shape=(1, 4)

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

    def add_recordings(self, sec, x=0.5):
        if not hasattr(self, 'recordings'):
            self.recordings = {}
        if 't' not in list(self.recordings.keys()):
            self.recordings['t'] = h.Vector().record(h._ref_t)
        tempname = sec.name() + '(' + str(x) + ')'
        self.recordings[tempname] = h.Vector().record(sec(x)._ref_v)

    def run_simulation_EFtest(self, Evar=0., tvar=0., dt=0.025, duration=0., theta=90, phi=90):
        '''Simulation'''
        h.calcesE(theta, phi)
        h.dt = dt
        h.tstop = duration
        h.setstim_snowp()
        h.stim_amp.from_python(Evar)
        h.stim_time.from_python(tvar)
        h.attach_stim()
        # neuron.h.stdinit()
        # neuron.h.cvode_active(1)
        neuron.h.run()

cell = L5PTcell()


'''EPSP-like current at the nexus'''
secidx = cell.allSectionNames.index('dendA1_0000000000001')
apicalIClamp = h.IClamp(cell.allSectionList[14](1))
apicalIClamp.delay  = 0.
apicalIClamp.dur    = 1e9
apicalIClamp.amp    = 0.
'''somatic input'''
somaIClamp = h.IClamp(cell.allSectionList[0](0.5))
somaIClamp.delay  = 0.
somaIClamp.dur    = 1e9
somaIClamp.amp    = 0.
'''define the recordings'''
cell.add_recordings(cell.allSectionList[0], x=0.5)
cell.add_recordings(cell.allSectionList[14], x=1.0)
cell.recordings['apicalIClamp_Iepsp'] = h.Vector().record(apicalIClamp._ref_i)
cell.recordings['apicalIClamp_Isoma'] = h.Vector().record(somaIClamp._ref_i)
cell.recordings['apicalICa'] = h.Vector().record(h.dendA1_0000000000001(1)._ref_ica)

recordingsKeysList = list(cell.recordings.keys())


def calcEffectOfTdcsOnCaSpike(paraImax, paraVds, paraTde):
    dt = 0.025
    DEL, DUR = 510, 140
    tstop = DEL + DUR + 0
    tvar = np.arange(0, tstop, dt)

    AMP_AC, FREQ = 0., 10
    AMP_DC = paraVds
    Evar_DEL = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR = AMP_DC + AMP_AC * np.sin(2 * np.pi * FREQ * np.arange(0, DUR, dt) / 1000)
    Evar_tstop = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))

    Iepsp = - generate_epsp_like_current(taur=0.8, taud=4, imax=paraImax, onset=DEL+paraTde, dt=dt, tstop=tstop)
    tvar_hoc = h.Vector().from_python(tvar)
    Iepsp_hoc = h.Vector().from_python(Iepsp)
    Iepsp_hoc.play(apicalIClamp._ref_amp, tvar_hoc, True)
    somaIClamp.delay = 0.
    somaIClamp.dur = 1e9
    somaIClamp.amp = 0.

    cell.run_simulation_EFtest(Evar=Evar, tvar=tvar, dt=dt, duration=tstop, theta=90, phi=90)

    idx1 = 25
    idx2 = int(tstop / dt)
    t = cell.recordings['t'].to_python()[idx1:idx2]
    vsoma = cell.recordings['somaA(0.5)'].to_python()[idx1:idx2]
    vdend = cell.recordings['dendA1_0000000000001(1.0)'].to_python()[idx1:idx2]
    Id = cell.recordings['apicalIClamp_Iepsp'].to_python()[idx1:idx2]
    Is = cell.recordings['apicalIClamp_Isoma'].to_python()[idx1:idx2]
    ICa = cell.recordings['apicalICa'].to_python()[idx1:idx2]
    t, vsoma, vdend = np.array(t), np.array(vsoma), np.array(vdend)
    Id, Is, ICa = np.array(Id), np.array(Is), np.array(ICa)

    CaPeakAmplitude = np.max(vdend)
    CaStartTime, CaEndTime, CaDuration = 0, 0, 0
    CaSpikeBool, ts, te = isCalciumSpike(t, vdend, CaSpike_startVth=-30, CaSpike_endVth=-50)
    if CaSpikeBool == 1:
        CaStartTime, CaEndTime, CaDuration = ts, te, te-ts

    spikeNumber, spikeTime = calcSpikeNumberAndSpikeTime(t, vsoma)
    firstSpikeTime, secondSpikeTime, firingRateWithinBurst = 0, 0, 0
    if spikeNumber > 0:
        firstSpikeTime = spikeTime[0]
    if spikeNumber > 1:
        secondSpikeTime = spikeTime[1]
        firingRateWithinBurst = 1e3 / np.mean(np.diff(spikeTime))

    ICaPeakAmplitude = np.abs(np.min(ICa))
    ICaBool, ts, te = calcICaTime(t, ICa, ICaVth=-0.12)
    ICaStartTime, ICaEndTime, ICaDuration = 0, 0, 0
    if ICaBool == 1:
        ICaStartTime, ICaEndTime, ICaDuration = ts, te, te - ts

    return paraImax, paraVds, paraTde, \
           CaStartTime, CaEndTime, CaDuration, CaPeakAmplitude, \
           spikeNumber, firstSpikeTime, secondSpikeTime, firingRateWithinBurst, \
           ICaStartTime, ICaEndTime, ICaDuration, ICaPeakAmplitude


def calcEffectOfTdcsOnCaSpike_EfieldDirection(paraImax, paraVds, paraTde, paraEtheta):
    dt = 0.025
    DEL, DUR = 510, 350
    tstop = DEL + DUR + 0
    tvar = np.arange(0, tstop, dt)

    AMP_AC, FREQ = 0., 10
    AMP_DC = paraVds
    Evar_DEL = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR = AMP_DC + AMP_AC * np.sin(2 * np.pi * FREQ * np.arange(0, DUR, dt) / 1000)
    Evar_tstop = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))

    Iepsp = - generate_epsp_like_current(taur=0.8, taud=4, imax=paraImax, onset=DEL+paraTde, dt=dt, tstop=tstop)
    tvar_hoc = h.Vector().from_python(tvar)
    Iepsp_hoc = h.Vector().from_python(Iepsp)
    Iepsp_hoc.play(apicalIClamp._ref_amp, tvar_hoc, True)
    somaIClamp.delay = 0.
    somaIClamp.dur = 1e9
    somaIClamp.amp = 0.

    cell.run_simulation_EFtest(Evar=Evar, tvar=tvar, dt=dt, duration=tstop, theta=paraEtheta, phi=90)

    idx1 = 25
    idx2 = int(tstop / dt)
    t = cell.recordings['t'].to_python()[idx1:idx2]
    vsoma = cell.recordings['somaA(0.5)'].to_python()[idx1:idx2]
    vdend = cell.recordings['dendA1_0000000000001(1.0)'].to_python()[idx1:idx2]
    Id = cell.recordings['apicalIClamp_Iepsp'].to_python()[idx1:idx2]
    Is = cell.recordings['apicalIClamp_Isoma'].to_python()[idx1:idx2]
    ICa = cell.recordings['apicalICa'].to_python()[idx1:idx2]
    t, vsoma, vdend = np.array(t), np.array(vsoma), np.array(vdend)
    Id, Is, ICa = np.array(Id), np.array(Is), np.array(ICa)

    CaPeakAmplitude = np.max(vdend)
    CaStartTime, CaEndTime, CaDuration = 0, 0, 0
    CaSpikeBool, ts, te = isCalciumSpike(t, vdend, CaSpike_startVth=-30, CaSpike_endVth=-50)
    if CaSpikeBool == 1:
        CaStartTime, CaEndTime, CaDuration = ts, te, te-ts

    spikeNumber, spikeTime = calcSpikeNumberAndSpikeTime(t, vsoma)
    firstSpikeTime, secondSpikeTime, firingRateWithinBurst = 0, 0, 0
    if spikeNumber > 0:
        firstSpikeTime = spikeTime[0]
    if spikeNumber > 1:
        secondSpikeTime = spikeTime[1]
        firingRateWithinBurst = 1e3 / np.mean(np.diff(spikeTime))

    ICaPeakAmplitude = np.abs(np.min(ICa))
    ICaBool, ts, te = calcICaTime(t, ICa, ICaVth=-0.12)
    ICaStartTime, ICaEndTime, ICaDuration = 0, 0, 0
    if ICaBool == 1:
        ICaStartTime, ICaEndTime, ICaDuration = ts, te, te - ts

    return paraImax, paraVds, paraTde, \
           CaStartTime, CaEndTime, CaDuration, CaPeakAmplitude, \
           spikeNumber, firstSpikeTime, secondSpikeTime, firingRateWithinBurst, \
           ICaStartTime, ICaEndTime, ICaDuration, ICaPeakAmplitude


def calcEffectOfTdcsOnCaSpike_tDE(paraImax, paraVds, paraTde, paraEtheta):
    dt = 0.025
    DEL, DUR = 510, 350
    tstop = DEL + DUR + 0
    tvar = np.arange(0, tstop, dt)

    AMP_AC, FREQ = 0., 10
    AMP_DC = paraVds
    Evar_DEL = np.zeros_like(np.arange(0, DEL, dt))
    Evar_DUR = AMP_DC + AMP_AC * np.sin(2 * np.pi * FREQ * np.arange(0, DUR, dt) / 1000)
    Evar_tstop = np.zeros_like(np.arange(0, tstop - DEL - DUR, dt))
    Evar = np.hstack((Evar_DEL, Evar_DUR, Evar_tstop))

    Iepsp = - generate_epsp_like_current(taur=0.8, taud=4, imax=paraImax, onset=DEL+paraTde, dt=dt, tstop=tstop)
    tvar_hoc = h.Vector().from_python(tvar)
    Iepsp_hoc = h.Vector().from_python(Iepsp)
    Iepsp_hoc.play(apicalIClamp._ref_amp, tvar_hoc, True)
    somaIClamp.delay = 0.
    somaIClamp.dur = 1e9
    somaIClamp.amp = 0.

    cell.run_simulation_EFtest(Evar=Evar, tvar=tvar, dt=dt, duration=tstop, theta=paraEtheta, phi=90)

    idx1 = 25
    idx2 = int(tstop / dt)
    t = cell.recordings['t'].to_python()[idx1:idx2]
    vsoma = cell.recordings['somaA(0.5)'].to_python()[idx1:idx2]
    vdend = cell.recordings['dendA1_0000000000001(1.0)'].to_python()[idx1:idx2]
    Id = cell.recordings['apicalIClamp_Iepsp'].to_python()[idx1:idx2]
    Is = cell.recordings['apicalIClamp_Isoma'].to_python()[idx1:idx2]
    ICa = cell.recordings['apicalICa'].to_python()[idx1:idx2]
    t, vsoma, vdend = np.array(t), np.array(vsoma), np.array(vdend)
    Id, Is, ICa = np.array(Id), np.array(Is), np.array(ICa)

    CaPeakAmplitude = np.max(vdend)
    CaStartTime, CaEndTime, CaDuration = 0, 0, 0
    CaSpikeBool, ts, te = isCalciumSpike(t, vdend, CaSpike_startVth=-30, CaSpike_endVth=-50)
    if CaSpikeBool == 1:
        CaStartTime, CaEndTime, CaDuration = ts, te, te-ts

    spikeNumber, spikeTime = calcSpikeNumberAndSpikeTime(t, vsoma)
    firstSpikeTime, secondSpikeTime, firingRateWithinBurst = 0, 0, 0
    if spikeNumber > 0:
        firstSpikeTime = spikeTime[0]
    if spikeNumber > 1:
        secondSpikeTime = spikeTime[1]
        firingRateWithinBurst = 1e3 / np.mean(np.diff(spikeTime))

    ICaPeakAmplitude = np.abs(np.min(ICa))
    ICaBool, ts, te = calcICaTime(t, ICa, ICaVth=-0.12)
    ICaStartTime, ICaEndTime, ICaDuration = 0, 0, 0
    if ICaBool == 1:
        ICaStartTime, ICaEndTime, ICaDuration = ts, te, te - ts

    return paraImax, paraVds, paraTde, \
           CaStartTime, CaEndTime, CaDuration, CaPeakAmplitude, \
           spikeNumber, firstSpikeTime, secondSpikeTime, firingRateWithinBurst, \
           ICaStartTime, ICaEndTime, ICaDuration, ICaPeakAmplitude

if __name__ == '__main__':
    start_time = time.time()
    casemark = 100
    '''Paper Figure 3'''
    if casemark == 1:
        list_Imax = [1.6]
        list_Vds = np.arange(-15, 15.1, 1)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcEffectOfTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        np.save('data/Imax1.6nA_tde20ms_varyVds-15-15mV.npy', res)

    '''Paper Figure 4(b-d)'''
    if casemark == 2:
        list_Imax = [3]
        list_Vds = np.arange(-15, 15.1, 1)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcEffectOfTdcsOnCaSpike, paras)
        pool.close()
        pool.join()
        np.save('data/Imax3nA_tde20ms_varyVds-15-15mV.npy', res)

    '''Paper Figure 4(e)'''
    if casemark == 3:
        # Runnig time is 7833 seconds
        list_Imax = np.arange(1, 4.01, 0.02)
        list_Vds = np.arange(-15, 15.1, 1)
        list_tde = [20]
        paras = [[i, j, k] for i in list_Imax for j in list_Vds for k in list_tde]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcEffectOfTdcsOnCaSpike, paras)
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
        np.save('data/Imax1-4nA_tde20ms_varyVds-15-15mV.npy', results)

    '''Paper Figure 5'''
    if casemark == 4:
        list_Imax = [1.6, 3]
        list_Vds = [-15, -10, -5, -2.5, 0, 2.5, 5, 10, 15]
        list_tde = np.arange(0, 300+1, 10)
        listEfieldDirection = [90]
        paras = [[i, j, k, m] for i in list_Imax for j in list_Vds for k in list_tde for m in listEfieldDirection]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcEffectOfTdcsOnCaSpike_tDE, paras)
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
        np.save('data/Imax1.6-3nA_tde0-300ms_Vds-15-15mV_EFDirection90degree.npy', results)

    '''Paper Figure 6'''
    if casemark == 5:
        list_Imax = [1.6, 3]
        list_Vds = [0, 2.5, 5]
        list_tde = [0, 10, 20, 250, 300]
        listEfieldDirection = np.arange(-90, 270+1, 15)
        paras = [[i, j, k, m] for i in list_Imax for j in list_Vds for k in list_tde for m in listEfieldDirection]
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(calcEffectOfTdcsOnCaSpike_EfieldDirection, paras)
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
        np.save('data/Imax1.6-3nA_tde0-10-20-250-300ms_Vds0-2.5-5mV_EFDirection-180-180degree.npy', results)

    print('Running time:', time.time() - start_time)