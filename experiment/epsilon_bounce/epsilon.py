import numpy as np


class BounceEpsilon(object):
    def __init__(self, init, final, loss, gravity):
        self._init = init
        self._final = final
        self._loss = loss
        self._gravity = gravity
        self._height = self._init - self._final
        self.reset()

    def reset(self):
        self._t = 0
        self._is_fall = True
        self._v = 0
        self._height = self._init - self._final
        self._stage_t = np.sqrt(2 * self._height / self._gravity)
        self._staged_t = self._stage_t
        self._mid_t = 0
        self._h_func = lambda: self._height - self._gravity * ((self._t - self._mid_t) ** 2) / 2

    def step(self):
        self._h = self._h_func()
        self._t += 1
        if self._t > self._staged_t:
            self._height *= self._loss
            self._stage_t = np.sqrt(2 * self._height / self._gravity)
            self._mid_t = self._staged_t + self._stage_t
            self._staged_t += 2 * self._stage_t
        return self._h


if __name__ == '__main__':
    be = BounceEpsilon(1.0, 0.01, 0.5, 0.01)
    heights = []
    for _ in range(1000):
        heights.append(be.step())
    print(heights)
