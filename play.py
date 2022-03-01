import wave
import numpy as np
from numba import njit
import pyglet

FILENAME = 'audio/stepsine.wav'

raw = wave.open(FILENAME, 'r')
audio = raw.readframes(-1)
audio = np.frombuffer(audio, dtype='int16')
if raw.getnchannels() == 2:
    audio = audio.reshape(-1, 2)
    audio = audio[:, 0]/(2**16)+audio[:, 1]/(2**16)
else:
    audio = audio/(2**15)
f_rate = raw.getframerate()

print(audio.shape)
print(f_rate)

window = pyglet.window.Window(500, 500)
batch = pyglet.graphics.Batch()

y_axis = pyglet.shapes.Line(250, 0, 250, 500, width=4, batch=batch)
x_axis = pyglet.shapes.Line(0, 250, 500, 250, width=4, batch=batch)

player = pyglet.media.Player()
source = pyglet.media.load(FILENAME, streaming=False)
player.queue(source)

@njit(fastmath=True, cache=True)
def process(audio_clip, n_samples, i, f_max):
    points = np.empty((n_samples, 4))
    
    # audio_clip = audio_clip/2+0.5

    DELTA_PHI = 2*np.pi/44100*f_max
    for j in range(n_samples):
        theta = (i-j)*DELTA_PHI
        f_theta = audio_clip[-(j+1)]

        points[j, 0] = 250*np.cos(theta)*f_theta+250
        points[j, 1] = 250*np.sin(theta)*f_theta+250
        if j != 0:
            points[j, 2] = points[j-1, 0]
            points[j, 3] = points[j-1, 1]
        else: # For now, just clone the coordinates so that this segment doesn't exist
            points[j, 2] = points[j, 0]
            points[j, 3] = points[j, 1]
    
    return points

class trace:
    def __init__(self, audio, n_samples):
        self.audio = audio
        self.f_rate = f_rate
        self.n_samples = n_samples
        self.lines = [
            pyglet.shapes.Line(
                250, 250, 250, 250, 
                width=2, 
                color=(int(255*np.exp(-i/n_samples)), 0, 0), 
                batch=batch) 
            for i in range(n_samples)
        ] 

        self.i = 0
    def update(self, dt):
        FFT_SIZE = 2**14 # increasing this reduces unintended rotation
        FFT_OUTPUT_SIZE = int(FFT_SIZE/2)
        if self.i > FFT_SIZE and self.i > self.n_samples:
            audio_clip = self.audio[self.i-FFT_SIZE:self.i]
            audio_clip_fft = np.fft.fft(audio_clip)[:FFT_OUTPUT_SIZE]
            k_max = np.argmax(np.abs(audio_clip_fft))
            f_max = k_max/FFT_OUTPUT_SIZE * (44100/2)
            # print(f_max)

            audio_clip = self.audio[self.i-self.n_samples:self.i]
            
            audio_clip = np.fft.fft(audio_clip)

            # making a new magnitude spectrum equal to energy of original mag 
            #  spectrum b.c. that will emphasize the most energetic frequency
            #  components
            magnitude = np.abs(audio_clip)
            new_magnitude = magnitude**2
            old_energy = np.sum(new_magnitude) # new_magnitude is just energy of mag spectrum
            new_energy = np.sum(new_magnitude**2)
            rescale_factor = np.sqrt(old_energy/new_energy)
            new_magnitude = new_magnitude*rescale_factor # rescaling to ensure energy remains unchanged
            
            # using original phase spectrum to build new signal
            audio_clip = new_magnitude*np.exp(1j*np.angle(audio_clip))
            audio_clip = np.real(np.fft.ifft(audio_clip))

            points = process(audio_clip, self.n_samples, self.i, f_max)

            for i in range(self.n_samples):
                self.lines[i].x = points[i, 0]
                self.lines[i].y = points[i, 1]
                self.lines[i].x2 = points[i, 2]
                self.lines[i].y2 = points[i, 3]
        # self.i += 441
        self.i = int(44100*player.time) # TODO: player is in global namespace...

signal_trace = trace(audio, 512) # At least 441 points are needed to be complete at 100fps!

@window.event
def on_draw():
    window.clear()
    batch.draw()

player.play()
pyglet.clock.schedule_interval(signal_trace.update, 1/100.0)
pyglet.app.run()
