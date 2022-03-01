# rose-music-visualizer

The rose curves on a polar plot look pretty cool, so I decided to apply this observation to music.

It uses an FFT to find what frequency has the most energy. Then, it plots samples from the music such that a pure tone of that frequency looks like a 1-petal rose (i.e. a circle). Consequently, any overtones becomes a certain k-petal rose. All of the graphics is performed on pyglet.

Right now, this program is a WIP.