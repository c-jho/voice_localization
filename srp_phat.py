import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, lfilter

SOUND_SPEED         = 343.2

class srp_phat:
    def __init__(self, channel, mic_dist, mic_pairs, mic_dim):
        self.MIC_DIST       = mic_dist
        self.mic_pairs      = mic_pairs
        self.MIC_DIM        = mic_dim 
        self.CHANNEL        = channel
        
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grid_point     = self.generate_hemisphere_grid()
        self.tdoa_lut       = self.generate_tdoa_table(self.grid_point)

        self.freqs          = np.fft.rfftfreq(16000, d=1/16000)
        self.valid_idx      = (self.freqs > 300) & (self.freqs < 3400)
        self.phase_shift    = self.calc_phase_shift()

        # self.sc, self.ax    = self.plot_init(self.grid_point)


    def calc_phase_shift(self):
        tau_expanded   = self.tdoa_lut[:, :, np.newaxis]
        
        self.freqs     = self.freqs[self.valid_idx]
        freqs_expanded = self.freqs[np.newaxis, np.newaxis, :]

        phase_shift    = np.exp(1j * 2 * np.pi * freqs_expanded * tau_expanded)

        phase_shift    = torch.from_numpy(phase_shift)
        phase_shift    = phase_shift.to(self.device)

        return phase_shift


    def generate_hemisphere_grid(self, azimuth_step=6, elevation_step=5):
        points = []
        count = 1
        step = azimuth_step

        for elev_deg in range(0, 91, elevation_step):
            elev_rad = np.radians(elev_deg)
            
            if elev_deg >= 80: azimuth_step += step*count; count += 1
            if elev_deg == 90: azimuth_step = 360

            for azim_deg in range(0, 360, azimuth_step):
                azim_rad = np.radians(azim_deg)

                x = np.cos(elev_rad)*np.cos(azim_rad)
                y = np.cos(elev_rad)*np.sin(azim_rad)
                z = np.sin(elev_rad)

                points.append([x,y,z])

        return np.array(points)


    def generate_tdoa_table(self, grid_point):

        expected_tdoa = np.zeros((len(grid_point), len(self.mic_pairs)))

        for point_idx, src_pos in enumerate(grid_point):
            for pair_idx, (i, j) in enumerate(self.mic_pairs):
                dist_i = np.linalg.norm(src_pos - self.MIC_DIM[i])
                dist_j = np.linalg.norm(src_pos - self.MIC_DIM[j])
                expected_tdoa[point_idx, pair_idx] = (dist_i - dist_j) / SOUND_SPEED
        
        return expected_tdoa

    
    def calc_srp_phat(self, signal, sample_rate=16000):
        sig = [0,0,0,0]
        for i in range(4): sig[i] = signal[i+1::self.CHANNEL]
        signal_ = np.array([sig[0],sig[1],sig[2],sig[3]])

        G = []
        n = signal_[0].shape[0] + signal_[1].shape[0]
        
        # 각 mic pair 별 주파수 영역 위상차 계산
        for pair_idx, (i, j) in enumerate(self.mic_pairs):
            Xi = np.fft.rfft(signal_[i], n=n)
            Xj = np.fft.rfft(signal_[j], n=n)
            R = Xi * np.conj(Xj)
            G.append(R / (np.abs(R) + 1e-10))
        G = np.array(G)

        # 사람 음성 주파수 영역대 필터링
        G = G[:, self.valid_idx]
        srp_power = np.zeros(self.tdoa_lut.shape[0])
        
        # torch 사용해서 계산
        G = torch.from_numpy(G[np.newaxis, :, :])
        G = G.to(self.device)
        aligned = G * self.phase_shift
        srp_power = torch.sum(torch.real(torch.sum(aligned, axis=2)), axis=1)
        ex_point = torch.argmax(srp_power)  # Power가 높은 index 추출

        # torch 없이 numpy로 계산
        # aligned = G[np.newaxis, :, :] * self.phase_shift
        # srp_power = np.sum(np.real(np.sum(aligned, axis=2)), axis=1)
        # ex_point = np.argmax(srp_power)   # Power가 높은 index 추출
        
        # real-time srp_power 그래프 update
        # self.update_plot(self.grid_point, srp_power.detach().cpu().numpy())

        # 해당 point의 degree 추출
        degree = np.degrees(np.arctan2(self.grid_point[ex_point, 1], self.grid_point[ex_point, 0]))
        degree = (degree + 360) % 360
        return degree
    
    def plot_init(self, gp):
        x = gp[:, 0]
        y = gp[:, 1]
        z = gp[:, 2]
        plt.ion()
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=np.zeros_like(x), cmap='jet', s=20)
        cb = fig.colorbar(sc, ax=ax, label='SRP Power')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("SRP-PHAT Real-Time")
        plt.tight_layout()

        return sc, ax
    
    def update_plot(self, gp, srp):
        x = gp[:, 0]
        y = gp[:, 1]
        z = gp[:, 2]
        self.sc.remove()
        self.sc = self.ax.scatter(x, y, z, c=srp, cmap='jet', s=20)
        plt.draw()
        plt.pause(0.005)
