import multiprocessing
import cv2 as cv
import numpy as np
from multiprocessing import Process


def video_loop(sound_value):
    cv.namedWindow("PRETTYLIGHTMAJIG", cv.WINDOW_NORMAL)
    cv.resizeWindow("PRETTYLIGHTMAJIG", 1920, 1080)

    cap = cv.VideoCapture(0)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0+int(sound_value.value), 150+int(sound_value.value), cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('PRETTYLIGHTMAJIG', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next
    cv.destroyAllWindows()


def sound_loop(sound_value):
    import sounddevice as sd

    def mic_to_int(indata, outdata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        sound_value.value = volume_norm

    with sd.Stream(callback=mic_to_int):
        sd.sleep(1)


if __name__ == '__main__':
    sound_value = multiprocessing.Value('i',0)
    video = Process(target=video_loop, args=[sound_value]).start()
    sound = Process(target=sound_loop, args=[sound_value]).start()
