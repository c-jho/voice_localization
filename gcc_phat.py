"""
 Estimate time delay using GCC-PHAT 
 Copyright (c) 2017 Yihui Xiong

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

#GCC_PHAT(Generalized Cross Correlation-Phase Transform)
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0] # chunk size 더함.

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)         # rFFT(real(실수) Fast Furier Transform)
    REFSIG = np.fft.rfft(refsig, n=n)   
    R = SIG * np.conj(REFSIG)           # np.conj (complex conjugate, 켤레복소수, 복소수 값의 부호 반대로.), 부호 반대로 한 후 곱하면 위상 차이값을 구할 수 있음.
    safe_div = np.where(np.abs(R) == 0, 0, R / np.abs(R))

    cc = np.fft.irfft(safe_div, n=(interp * n))    # abs(R)을 나누어 실수값을 제거하여 위상값(복소수)만 남겨두고 역변환하여 두 신호의 샘플간의 시간차 값을 표현
    cc1 = cc
    max_shift = int(interp * n / 2) # 128000
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)   # 보간(interpolation)비율 * fs * max_tau = 마이크 거리에 따른 최대 샘플 이동량 * 보간비율값

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))    
    # 기존에 irfft를 통해 나온 데이터는 원형구조로 만약 2000개의 n이면 index가 0~1000:0~1000, 1001~2000:-1000~-1 형태여서 이를 max shift값 범위내 데이터만 순차적으로 정렬

    # cc의 데이터는 시간지연값임. 이때 가장 큰 시간지연이 있는 위치와 max_shift간의 차는 지연된 시간동안의 샘플수 * 보간비율 된 값.
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)    # 보간값과 초당 sample수를 나누면 지연시간을 구할 수 있음.

    return tau, cc


def main():
    
    refsig = np.linspace(1, 10, 10)

    for i in range(0, 10):
        sig = np.concatenate((np.linspace(0, 0, i), refsig, np.linspace(0, 0, 10 - i)))
        offset, _ = gcc_phat(sig, refsig, visualize=True)
        print(offset)


if __name__ == "__main__":
    main()




