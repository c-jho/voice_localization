import sys
import queue
import math, time
import wave
import pyaudio
import threading
import numpy as np
from collections import deque
from gcc_phat import gcc_phat
from matplotlib import pyplot as plt
from silero_vad import load_silero_vad, get_speech_timestamps


CHANNEL             = 6
RATE                = 16000                                 # 초당 샘플 수
CHUNK_SIZE          = 400                                   # 25ms
SOUND_SPEED         = 343.2                                 # m/s
MIC_DISTANCE_4      = 0.06463                               # 대각 방향의 마이크 거리(m)
MAX_TDOA_4          = MIC_DISTANCE_4 / float(SOUND_SPEED)   # 대각방향의 마이크 입력 시간차
VAD_MAX_CHUNK_SIZE  = 8000

class MicArray(object):
    def __init__(self, rate=16000, channels=6, chunk_size=None):
        self.pyaudio_instance   = pyaudio.PyAudio()
        self.dqueue             = deque(maxlen=(int)(VAD_MAX_CHUNK_SIZE/CHUNK_SIZE))
        self.lock               = threading.Lock()
        self.quit_event         = threading.Event()
        self.channels           = channels
        self.sample_rate        = rate
        self.chunk_size         = chunk_size if chunk_size else rate / 100
        self.graph_data         = []
        self.queue_input_flag   = False
        self.rms_chunk_size     = 800
        self.threshold          = 2.5

        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):               #PC에 연결된 마이크 장치 read.
            dev = self.pyaudio_instance.get_device_info_by_index(i)             #마이크 장치 번호 저장
            name = dev['name'].encode('utf-8')                                  #마이크 장치 이름 저장
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])   
            if dev['maxInputChannels'] == self.channels:                        #사용할 마이크 channel수와 동일한 장치 확인
                print('Use {}'.format(name))
                device_index = i                                                #사용할 마이크 장치번호 저장
                break

        if device_index is None:
            raise Exception('can not find input device with {} channel(s)'.format(self.channels))

        self.stream = self.pyaudio_instance.open(   # 마이크 장치 연결
            input=True,                             # 마이크 입력 True/False
            start=False,                            # stream을 열때 자동으로 시작할지 True/False(start로 수동제어 필요)
            format=pyaudio.paInt16,                 # 마이크 샘플 형식 (Int16 = 16비트 정수(-32768~32767, 2bytes))
            channels=self.channels,                 # 사용할 channel 수
            rate=int(self.sample_rate),             # 초당 샘플 수
            frames_per_buffer=int(self.chunk_size), # 채널 당 한번 읽을 때 받을 chunk size
            stream_callback=self._callback,         # 마이크 입력 받을 때 호출될 callback 함수
            input_device_index=device_index,        # 입력 장치 번호
        )

    def _callback(self, in_data, frame_count, time_info, status):
        with self.lock:
            self.dqueue.append(in_data)
        return None, pyaudio.paContinue
    
    def start(self):
        self.dqueue.clear()
        self.stream.start_stream()

    def rms_filter(self, chunk):
        ck = np.array(chunk, copy=True)
        rms = []
        for j in range(0, len(ck)-self.rms_chunk_size, self.rms_chunk_size):
            rms.append(np.sqrt(np.mean(np.square(ck[j:j+self.rms_chunk_size].astype('int32')))))
        rms_mean = np.mean(rms)

        for j in range(len(rms)):
            if rms[j] > rms_mean*self.threshold:
                start_idx = j * self.rms_chunk_size
                ck[start_idx:start_idx + self.rms_chunk_size] = np.zeros(self.rms_chunk_size, dtype=ck.dtype)

        return ck

    def read_chunks(self):
        self.quit_event.clear()

        while not self.quit_event.is_set():
            if(len(self.dqueue) > 0):
                frames = b''.join(self.dqueue)

                frames = np.frombuffer(frames, dtype='int16')
                yield frames
            else: pass
    
    def save_audio(self, file_name, data):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, type, value, traceback):
        if value:
            return False
        self.stop()

    def get_direction(self, buf):
        best_guess = None
        if self.channels == CHANNEL:
            MIC_GROUP_N = 2
            MIC_GROUP = [[1, 3], [2, 4]]

            tau = [0] * MIC_GROUP_N
            theta = [0] * MIC_GROUP_N
            
            for i, v in enumerate(MIC_GROUP):
                #rms filtering & calculate TDOA
                # tau[i], _ = gcc_phat(self.rms_filter(buf[v[0]::CHANNEL]), self.rms_filter(buf[v[1]::CHANNEL]), fs=self.sample_rate, max_tau=MAX_TDOA_4, interp=32)
                
                # only calculate TDOA
                tau[i], _ = gcc_phat(buf[v[0]::CHANNEL], buf[v[1]::CHANNEL], fs=self.sample_rate, max_tau=MAX_TDOA_4, interp=32)
                
                theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180 / math.pi
                # print(theta[i])
                    
            if np.abs(theta[0]) < np.abs(theta[1]):
                if theta[1] > 0:
                    best_guess = (theta[0] + 360) % 360
                else:
                    best_guess = (180 - theta[0])
            else:
                if theta[0] < 0:
                    best_guess = (theta[1] + 360) % 360
                else:
                    best_guess = (180 - theta[1])

                best_guess = (best_guess + 90 + 180) % 360
            
            best_guess = (best_guess + 45) % 360
            best_guess = (360 - best_guess) % 360
            return best_guess



def main():
    import signal

    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print('Quit')
    signal.signal(signal.SIGINT, signal_handler)
    silero_vad_model = load_silero_vad()
    
    with MicArray(RATE, CHANNEL, CHUNK_SIZE) as mic:
        for chunk in mic.read_chunks():
            vad_audio_data1 = chunk[0::CHANNEL]
            vad_audio_data = vad_audio_data1.astype(np.float32) / 32768.0
            vad_audio_data = np.array(vad_audio_data, dtype=np.float32, copy=True)
            speech_timestamps = get_speech_timestamps(vad_audio_data, silero_vad_model)

            if len(speech_timestamps) > 0:
                direction = mic.get_direction(chunk)
                print("degree : ",int(direction))

            if is_quit.is_set():
                break


if __name__ == '__main__':
    main()
