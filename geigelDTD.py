import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import resample

# =========================
# States
# =========================
IDLE = 0
FAR_ONLY = 1
NEAR_ONLY = 2
DOUBLE_TALK = 3

STATE_LABELS = ["idle", "far", "near", "double"]


# =========================
# Hard Geigel DTD
# =========================
class HardGeigelDTD:
    def __init__(self, fs, gf=2.0, eps=1e-12):
        self.fs = fs
        self.gf = gf
        self.eps = eps

        self.fast_a = np.exp(-1/(0.005*fs))
        self.slow_a = np.exp(-1/(0.250*fs))
        self.far_a  = np.exp(-1/(0.050*fs))
        self.gain_a = np.exp(-1/(1.0*fs))

        self.near_fast = eps
        self.near_slow = eps
        self.far_power = eps
        self.echo_gain = 1.0

        self.near_thresh = 0.02**2
        self.far_thresh  = 0.02**2

    def step(self, near, far):
        pn = near*near
        pf = far*far

        self.near_fast = self.fast_a*self.near_fast + (1-self.fast_a)*pn
        self.near_slow = self.slow_a*self.near_slow + (1-self.slow_a)*pn
        self.far_power = self.far_a*self.far_power + (1-self.far_a)*pf

        far_active  = self.far_power > self.far_thresh
        near_active = self.near_fast > self.near_thresh

        echo_est = self.echo_gain * self.far_power
        thresh   = self.gf * echo_est
        diff     = self.near_fast - thresh

        if not far_active:
            state = NEAR_ONLY if near_active else IDLE
        else:
            state = DOUBLE_TALK if diff > 0 else FAR_ONLY

        # update gain only in far-only
        if state == FAR_ONLY:
            ratio = self.near_slow / (self.far_power + self.eps)
            self.echo_gain = self.gain_a*self.echo_gain + (1-self.gain_a)*ratio

        return {
            "state": state,
            "dtd": 1.0 if state == FAR_ONLY else 0.0,
            "near_fast": self.near_fast,
            "near_slow": self.near_slow,
            "far_power": self.far_power,
            "threshold": thresh,
        }


# =========================
# Hybrid (soft gate)
# =========================
class HybridGeigelDTD(HardGeigelDTD):
    def __init__(self, fs):
        super().__init__(fs)

        self.attack_a  = np.exp(-1/(0.010*fs))
        self.release_a = np.exp(-1/(0.100*fs))
        self.gate = 0.0

    def step(self, near, far):
        out = super().step(near, far)

        target = 1.0 if out["state"] == FAR_ONLY else 0.0

        if target < self.gate:
            a = self.attack_a
        else:
            a = self.release_a

        self.gate = a*self.gate + (1-a)*target

        out["dtd"] = self.gate
        return out


# =========================
# Test Signal (all states)
# =========================
def make_test(fs,scenario=1):
    if scenario == 1:
        t = np.arange(0, 16.0, 1/fs)

        far = np.zeros_like(t)
        near = np.zeros_like(t)

        # far only
        far[int(1*fs):int(2*fs)] = 0.25*np.sin(2*np.pi*300*t[int(1*fs):int(2*fs)])

        # near only
        near[int(2*fs):int(3*fs)] = 0.25*np.sin(2*np.pi*600*t[int(2*fs):int(3*fs)])

        # double talk
        far[int(3*fs):int(4*fs)]  = 0.25*np.sin(2*np.pi*300*t[int(3*fs):int(4*fs)])
        near[int(3*fs):int(4*fs)] = 0.25*np.sin(2*np.pi*600*t[int(3*fs):int(4*fs)])

        # far only
        far[int(6*fs):int(7*fs)] = 0.25*np.sin(2*np.pi*300*t[int(6*fs):int(7*fs)])
        # near only
        near[int(8*fs):int(9*fs)] = 0.25*np.sin(2*np.pi*600*t[int(8*fs):int(9*fs)])
        # double talk
        far[int(10*fs):int(11*fs)]  = 0.25*np.sin(2*np.pi*300*t[int(10*fs):int(11*fs)])
        near[int(10*fs):int(11*fs)] = 0.25*np.sin(2*np.pi*600*t[int(10*fs):int(11*fs)])


        echo = 0.5 * far
        mic = echo + near + 0.005*np.random.randn(len(t))
    elif scenario == 2:
        Fs,farend_data = read('data\\Hill_noisy.wav')
#        farend_data = farend_data[Fs * 1:Fs * 2] / 32768.0  # limit to 10 seconds
        far = farend_data / 32768.0
        if Fs != fs:
            far = resample(far,int(len(far)*fs/Fs))
        Fs2,nearend_data = read('data\\armstrong_noisy.wav')
        mic = nearend_data / 32768.0
        if Fs2 != fs:
            mic = resample(mic,int(len(mic)*fs/Fs2))
        t = np.arange(0, len(far))/fs

    return t, far, mic

def run_scenario(fs,scenario=1):
    t, far, mic = make_test(fs,scenario)
    hard = HardGeigelDTD(fs)
    hybrid = HybridGeigelDTD(fs)

    N = len(t)

    hard_dtd = np.zeros(N)
    hybrid_dtd = np.zeros(N)
    state = np.zeros(N)

    near_fast = np.zeros(N)
    near_slow = np.zeros(N)
    threshold = np.zeros(N)

    for n in range(N):
        h = hard.step(mic[n], far[n])
        s = hybrid.step(mic[n], far[n])

        hard_dtd[n] = h["dtd"]
        hybrid_dtd[n] = s["dtd"]
        state[n] = h["state"]

        near_fast[n] = h["near_fast"]
        near_slow[n] = h["near_slow"]
        threshold[n] = h["threshold"]

    # =========================
    # PLOTS (VERTICAL SUBPLOTS)
    # =========================
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Plot 1: signals + DTD ---
    axs[0].plot(t, far, label="far", alpha=0.5)
    axs[0].plot(t, mic, label="mic", alpha=0.5)

    axs[0].plot(t, hard_dtd, "--", label="hard DTD")
    axs[0].plot(t, hybrid_dtd, label="hybrid gate")

    axs[0].set_title("Signals + DTD Output")
    axs[0].legend()
    axs[0].grid()

    # --- Plot 2: envelopes ---
    axs[1].plot(t, np.sqrt(near_fast), label="near fast")
    axs[1].plot(t, np.sqrt(near_slow), label="near slow")
    axs[1].plot(t, np.sqrt(near_slow), label="far slow")
    axs[1].plot(t, np.sqrt(threshold), label="threshold")

    axs[1].set_title("Power Envelopes")
    axs[1].legend()
    axs[1].grid()

    # --- Plot 3: state ---
    axs[2].step(t, state, where="post")
    axs[2].set_yticks([0, 1, 2, 3])
    axs[2].set_yticklabels(STATE_LABELS)
    axs[2].set_title("DTD State")
    axs[2].grid()

    plt.tight_layout()

    plt.pause(.1)
    #plt.show()
    plt.pause(.1)


# =========================
# Harness
# =========================
def run():
    fs = 16000
    run_scenario(fs,1)
    run_scenario(fs,2)
    plt.show()  # pause



if __name__ == "__main__":
    run()