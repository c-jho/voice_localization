import sys
import argparse
import pyaudio
import threading
import numpy as np
from collections import deque
from gcc_phat import gcc_phat
from srp_phat import srp_phat
# from dxl_controller import dxl_controller
from matplotlib import pyplot as plt
from silero_vad import load_silero_vad, get_speech_timestamps

CHANNEL             = 6
MIC_DISTANCE_4      = 0.06463                               # 대각 방향의 마이크 거리(m)
CHUNK_SIZE          = 400                                   
VAD_MAX_CHUNK_SIZE  = 8000
RATE                = 16000                                 # 초당 샘플 수

# SRP-PHAT
MIC_DIST        = 0.02285
MIC_PAIRS       = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
MIC_DIM         = [ [MIC_DIST,MIC_DIST,0],
                    [-MIC_DIST,MIC_DIST,0],
                    [-MIC_DIST,-MIC_DIST,0],
                    [MIC_DIST,-MIC_DIST,0] ]

GCC             = 'gcc'
SRP             = 'srp'

class MicArray(object):
    def __init__(self, rate=16000, channels=6, chunk_size=None):
        self.pyaudio_instance   = pyaudio.PyAudio()
        self.dqueue             = deque(maxlen=(int)(VAD_MAX_CHUNK_SIZE/CHUNK_SIZE))
        self.lock               = threading.Lock()
        self.quit_event         = threading.Event()
        self.channels           = channels
        self.sample_rate        = rate
        self.chunk_size         = chunk_size if chunk_size else rate / 100

        # self.dxl                = dxl_controller('/dev/ttyU2D2')

        device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):               #PC에 연결된 마이크 장치 read.
            dev = self.pyaudio_instance.get_device_info_by_index(i)             #마이크 장치 번호 저장
            name = dev['name'].encode('utf-8')                                  #마이크 장치 이름 저장
            # print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])   
            if dev['maxInputChannels'] == self.channels:                        #사용할 마이크 channel수와 동일한 장치 확인
                print('\nUse {}'.format(name))
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

    def read_chunks(self):
        self.quit_event.clear()

        while not self.quit_event.is_set():
            if(len(self.dqueue) > 0):
                frames = b''.join(self.dqueue)
                frames = np.frombuffer(frames, dtype='int16')
                yield frames
            else: pass

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()
        # self.dxl.close_dxl()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, type, value, traceback):
        if value:
            return False
        self.stop()


def main(DOA_MODE):
    import signal
    
    is_quit = threading.Event()
    def signal_handler(sig, num):
        is_quit.set()
        print('Quit')
    signal.signal(signal.SIGINT, signal_handler)

    silero_vad_model = load_silero_vad()
    
    if DOA_MODE == GCC : gcc_phat_ = gcc_phat(CHANNEL, MIC_DISTANCE_4)
    elif DOA_MODE == SRP : srp_phat_ = srp_phat(CHANNEL, MIC_DIST, MIC_PAIRS, MIC_DIM)
    
    with MicArray(RATE, CHANNEL, CHUNK_SIZE) as mic:
        print(f"DOA_MODE : {DOA_MODE}_phat")
        print(f"___________________")

        for chunk in mic.read_chunks():
            vad_audio_data1 = chunk[0::CHANNEL]
            vad_audio_data = vad_audio_data1.astype(np.float32) / 32768.0
            vad_audio_data = np.array(vad_audio_data, dtype=np.float32, copy=True)
            speech_timestamps = get_speech_timestamps(vad_audio_data, silero_vad_model)

            if len(speech_timestamps) > 0:
                import time
                start = time.time()
                if DOA_MODE == GCC : degree = gcc_phat_.calc_gcc_phat(chunk)
                elif DOA_MODE == SRP : degree = srp_phat_.calc_srp_phat(chunk)
                # mic.dxl.dxl_control_pos((int)(degree*4095/360))
                print("deg:{:.1f}".format(degree), "| calc_time:{:.3f}".format(time.time() - start))

            if is_quit.is_set():
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doa', type=str, default='gcc', help='DOA 모드 선택(srp, gcc)')
    args = parser.parse_args()
    main(args.doa)
