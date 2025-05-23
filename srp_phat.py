import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SOUND_SPEED         = 343.2

class srp_phat:
    def __init__(self, channel, mic_dist, mic_pairs, mic_dim):
        self.MIC_DIST       = mic_dist
        self.mic_pairs      = mic_pairs
        self.MIC_DIM        = mic_dim 
        self.CHANNEL        = channel

        self.grid_point     = self.generate_hemisphere_grid()
        self.tdoa_lut       = self.generate_tdoa_table(self.grid_point)
        

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
        n = signal_[0].shape[0]
        freqs = np.fft.rfftfreq(n, d=1/sample_rate)

        # 각 mic pair 별 주파수 영역 위상차 계산
        for pair_idx, (i, j) in enumerate(self.mic_pairs):
            Xi = np.fft.rfft(signal_[i], n=n)
            Xj = np.fft.rfft(signal_[j], n=n)
            R = Xi * np.conj(Xj)
            G.append(R / (np.abs(R) + 1e-10))
        G = np.array(G)

        # 사람 음성 주파수 영역대 필터링
        valid_idx = (freqs > 300) & (freqs < 4000)
        freqs = freqs[valid_idx]
        G = G[:, valid_idx]

        srp_power = np.zeros(self.tdoa_lut.shape[0])
        
        # 벡터화 하여 SRP 계산
        tau_expanded = self.tdoa_lut[:, :, np.newaxis]                      # (P, M, 1)
        freqs_expanded = freqs[np.newaxis, np.newaxis, :]                   # (1, 1, F)
        phase_shift = np.exp(1j * 2 * np.pi * freqs_expanded * tau_expanded)# (P, M, F)
        aligned = G[np.newaxis, :, :] * phase_shift                         # (P, M, F)
        srp_power = np.sum(np.real(np.sum(aligned, axis=2)), axis=1)        # (P,)
        
        # Power가 높은 index 추출
        ex_point = np.argmax(srp_power)

        # 해당 point의 degree 추출
        degree = np.degrees(np.arctan2(self.grid_point[ex_point, 1], self.grid_point[ex_point, 0]))
        degree = (degree + 360) % 360
        return degree
