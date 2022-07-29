from bluepyopt.ephys.stimuli import LFPStimulus
from bluepyopt.ephys.recordings import Recording
from bluepyopt.ephys.responses import Response

import LFPy
import numpy as np
import pandas

import logging

logger = logging.getLogger(__name__)

# define default params

default_ecode_params = {
    "IDthres": {
        'duration': 270,
        'from': 0.5,
        'to': 1.3,
        'step': 0.04,
    },
    "firepattern": {
        'duration': 3600,
        'from': 1.2,
        'to': 2,
        'step': 0.8,
    },
    "IV": {
        'duration': 3000,
        'from': -1.4,
        'to': 0.6,
        'step': 0.2,
    },
    "IDrest": {
        'duration': 1350,
        'from': 0.5,
        'to': 3,
        'step': 0.25,
    },
    "APWaveform": {
        'duration': 50,
        'from': 2,
        'to': 3.5,
        'step': 0.3,
    },
    "HyperDepol": {
        'hyper_duration': 450,
        'hyper_from': -0.4,
        'hyper_to': -1.6,
        'hyper_step': -0.4,
        'depol_duration': 270,
        'depol_amp': 1
    },
    "sAHP": {
        'phase1_duration': 250,
        'phase1_amp': 0.4,
        'phase2_duration': 225,
        'phase2_from': 1.5,
        'phase2_to': 3,
        'phase2_step': 0.5,
        'phase3_duration': 450,
        'phase3_amp': 0.4
    },
    "PosCheops": {
        'amp': 3,
        'duration1': 4000,
        'duration2': 2000,
        'duration3': 1330,
        'delay': 1500,
    },
}


class sAHP(LFPStimulus):
    """sAHP current clamp injection"""

    def __init__(self,
                 holding_amplitude=0.0,
                 delay=250.0,
                 tmid=500.0,
                 tmid2=725.0,
                 toff=1175.0,
                 total_duration=1425.0,
                 phase1_amplitude=None,
                 phase2_amplitude=None,
                 phase3_amplitude=None,
                 location=None):
        """Constructor

        Args:
            holding_amplitude : holding potential (mV)
            delay (float): time to longstep of sahp (ms)
            tmid : time to second delay (ms) 
            tmid2: time to second delay being off (ms) 
            toff (float): amplitude at end of sahp (nA)
            total_duration (float): total duration (ms)
            phase1_amplitude (float): amplitude of phase1 (nA)
            phase1_amplitude (float): amplitude of phase2 (nA)
            phase1_amplitude (float): amplitude of phase3 (nA)
            location (Location): stimulus Location
        """

        super().__init__()
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.tmid = tmid
        self.tmid2 = tmid2
        self.toff = toff
        self.total_duration = total_duration
        self.phase1_amplitude = phase1_amplitude
        self.phase2_amplitude = phase2_amplitude
        self.phase3_amplitude = phase3_amplitude
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        # create vector to store the times at which stim amp changes
        times = sim.neuron.h.Vector()
        # create vector to store to which stim amps over time
        amps = sim.neuron.h.Vector()

        # at time 0.0, current is holding_amplitude
        times.append(0.0)
        amps.append(self.holding_amplitude)

        # until time delay, current is holding_amplitude
        times.append(self.delay)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude + self.phase1_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.phase1_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.phase2_amplitude)

        times.append(self.tmid2)
        amps.append(self.holding_amplitude + self.phase2_amplitude)

        times.append(self.tmid2)
        amps.append(self.holding_amplitude + self.phase3_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude + self.phase3_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude)

        times.append(self.total_duration)
        amps.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        # play the above current amplitudes into the current clamp
        amps.play(stim._ref_amp, times, 1)  # pylint: disable=W0212

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""

        # Destroy all persistent objects
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "sAHP phase 1 - %f phase 2  %f phase 3 %f  - delay %f mid delay %f  final delay %f" \
               "totdur %f at %s" % (
                   self.phase1_amplitude,
                   self.phase2_amplitude,
                   self.phase3_amplitude,
                   self.delay,
                   self.tmid,
                   self.tmid2,
                   self.total_duration,
                   self.location)


class HyperDepol(LFPStimulus):
    """HyperDepol Protocol"""

    def __init__(self,
                 holding_amplitude=0.0,
                 hyperpol_amplitude=None,
                 depol_amplitude=None,
                 delay=250.0,
                 tmid=700.0,
                 toff=970.0,
                 total_duration=1220.0,
                 location=None):
        """Constructor

        Args:
            amplitude (float): amplitude (nA)
            location (Location): stimulus Location
        """
        super().__init__()
        self.hyperpol_amplitude = hyperpol_amplitude
        self.depol_amplitude = depol_amplitude
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.tmid = tmid
        self.toff = toff
        self.total_duration = total_duration
        self.location = location

        # for efeature
        self.stim_start = self.tmid
        self.stim_end = self.toff
        self.step_amplitude = self.depol_amplitude

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        amps = sim.neuron.h.Vector()
        times = sim.neuron.h.Vector()

        times.append(0.0)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude)

        times.append(self.delay)
        amps.append(self.holding_amplitude + self.hyperpol_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.hyperpol_amplitude)

        times.append(self.tmid)
        amps.append(self.holding_amplitude + self.depol_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude + self.depol_amplitude)

        times.append(self.toff)
        amps.append(self.holding_amplitude)

        times.append(self.total_duration)
        amps.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        amps.play(
            stim._ref_amp,  # pylint:disable=W0212
            times,
            1
        )

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "HyperDepol with hyperopolarization amp %f and depolarization amplitude of %f at %s" % (
            self.hyperpol_amplitude,
            self.depol_amplitude,
            self.location)


class PosCheops(LFPStimulus):
    """PosCheops Protocol"""

    def __init__(self,
                 delay=250.0,
                 holding_amplitude=0,
                 t1=8250.,
                 t2=10250.,
                 t3=14250.,
                 t4=16250.,
                 toff=18910.,
                 total_duration=20910.0,
                 ramp1_amp=None,
                 ramp2_amp=None,
                 ramp3_amp=None,
                 location=None):
        """Constructor

        Args:
            TODO: CHANGE THESE
            amplitude (float): amplitude (nA)
            location (Location): stimulus Location
        """
        super().__init__()
        self.holding_amplitude = holding_amplitude
        self.delay = delay
        self.holding_amplitude = holding_amplitude
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.toff = toff
        self.total_duration = total_duration

        self.ramp1_amp = ramp1_amp
        self.ramp2_amp = ramp2_amp
        self.ramp3_amp = ramp3_amp

        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        ramp1_dur = (self.t1 - self.delay) / 2
        ramp2_dur = (self.t3 - self.t2) / 2
        ramp3_dur = (self.toff - self.t4) / 2

        current_vec = sim.neuron.h.Vector()
        time_vec = sim.neuron.h.Vector()

        time_vec.append(0.0)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.delay + ramp1_dur)
        current_vec.append(self.holding_amplitude + self.ramp1_amp)

        time_vec.append(self.t1)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.t2)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.t2 + ramp2_dur)
        current_vec.append(self.holding_amplitude + self.ramp2_amp)

        time_vec.append(self.t3)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.t4)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.t4 + ramp3_dur)
        current_vec.append(self.holding_amplitude + self.ramp3_amp)

        time_vec.append(self.toff)
        current_vec.append(self.holding_amplitude)

        time_vec.append(self.total_duration)
        current_vec.append(self.holding_amplitude)

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)
        stim = LFPyCell._hoc_stimlist[0]

        current_vec.play(
            stim._ref_amp,  # pylint:disable=W0212
            time_vec,
            1
        )

        # Make sure the following objects survive after instantiation
        self.persistent.append(time_vec)
        self.persistent.append(current_vec)

    def destroy(self, sim=None):
        """Destroy stimulus"""
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "PosCheops with ramp1_amp %f - ramp1_dur %f, ramp2_amp %f - ramp2_dur %f, and ramp3_amp %f " \
               "- ramp3_dur %fat %s" % (
            self.ramp1_amp,
            (self.t1 - self.delay) / 2,
            self.ramp2_amp,
            (self.t3 - self.de2lay) / 2,
            self.ramp3_amp,
            (self.toff - self.t4) / 2,
            self.location)


class NoiseOU3(LFPStimulus):
    """NoiseOU3 injection"""

    def __init__(self,
                 filename=None,
                 total_duration=1000.0,
                 location=None):
        """Constructor

        Args:
            filename (string): filepath for Noise
            location (Location): stimulus Location
        """

        super(NoiseOU3, self).__init__()
        self.filename = filename
        self.total_duration = total_duration
        self.location = location

        self.iclamp = None
        self.persistent = []

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        """Run stimulus"""
        from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation

        if hasattr(self.location, 'sec_index'):
            sec_index = self.location.sec_index
        elif isinstance(self.location, NrnSomaDistanceCompLocation):
            # compute sec_index closest to soma_distance
            cell_seg_locs = np.array([LFPyCell.xmid, LFPyCell.ymid, LFPyCell.zmid]).T
            soma_loc = LFPyCell.somapos
            dist_from_soma = np.array([np.linalg.norm(loc - soma_loc) for loc in cell_seg_locs])
            sec_index = np.argmin(np.abs(dist_from_soma - self.location.soma_distance))
        else:
            raise NotImplementedError(f"{type(self.location)} is currently not implemented with the LFPy backend")

        noisearray = np.loadtxt(self.filename, delimiter="\t", skiprows=1)

        # create vector to store the times at which stim amp changes
        times = sim.neuron.h.Vector(noisearray[:, 0])
        # create vector to store to which stim amps over time
        amps = sim.neuron.h.Vector(noisearray[:, 1])

        # amps = noisearray[:,1]
        # times = noisearray[:,0]

        self.iclamp = LFPy.StimIntElectrode(cell=LFPyCell,
                                            idx=sec_index,
                                            pptype='IClamp',
                                            dur=self.total_duration,
                                            record_current=True)

        stim = LFPyCell._hoc_stimlist[0]

        # play the above current amplitudes into the current clamp
        amps.play(stim._ref_amp, times, 1)  # pylint: disable=W0212

        # Make sure the following objects survive after instantiation
        self.persistent.append(times)
        self.persistent.append(amps)

    def destroy(self, sim=None):
        """Destroy stimulus"""

        # Destroy all persistent objects
        self.persistent = []
        self.iclamp = None

    def __str__(self):
        """String representation"""

        return "NoiseOU3 from %s located at %s" % (
            self.filename,
            self.location)


#### Stim response and recording ###
class TimeCurrentResponse(Response):

    """Response to stimulus"""

    def __init__(self, name, time=None, current=None):
        """Constructor

        Args:
            name (str): name of this object
            time (list of floats): time series
            current (list of floats): current series
        """

        super(TimeCurrentResponse, self).__init__(name)

        self.response = pandas.DataFrame()
        self.response["time"] = pandas.Series(time)
        self.response["current"] = pandas.Series(current)

    def read_csv(self, filename):
        """Load response from csv file"""

        self.response = pandas.read_csv(filename)

    def to_csv(self, filename):
        """Write response to csv file"""

        self.response.to_csv(filename)

    def __getitem__(self, index):
        """Return item at index"""

        return self.response.__getitem__(index)

    # This plot has to be generalised to several subplots
    def plot(self, axes):
        """Plot the response"""

        axes.plot(
            self.response["time"],
            self.response["current"],
            label="%s" % self.name,
        )


class StimRecording(Recording):

    """Stimulus response"""

    location = "stimulus"
    variable = "i"

    def __init__(self, name=None):
        """Constructor

        Args:
            name (str): name of this object
        """

        super(StimRecording, self).__init__(name=name)

        self.cell = None
        self.tvector = None
        self.time = None

        self.instantiated = False

    @property
    def response(self):
        """Return recording response"""

        if not self.instantiated:
            return None
        self.tvector = self.cell.tvec
        # if len(self.cell.pointprocesses) > 1:
        #     raise Exception
        return TimeCurrentResponse(
            self.name, self.tvector, self.cell.pointprocesses[0].i
        )

    def instantiate(self, sim=None, icell=None, LFPyCell=None):
        import LFPy

        """Instantiate recording"""

        logger.debug(
            "Adding recording of %s at %s", self.variable, self.location
        )

        assert isinstance(
            LFPyCell, LFPy.Cell
        ), "LFPRecording is only available for LFPCellModel"
        self.cell = LFPyCell
        self.tvector = None
        self.instantiated = True

    def destroy(self, sim=None):
        """Destroy recording"""
        self.tvector = None
        self.instantiated = False

    def __str__(self):
        """String representation"""

        return "%s: %s at %s" % (self.name, self.variable, self.location)
