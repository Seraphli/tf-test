import numpy as np


class BounceEpsilon(object):
    def __init__(self, init, final, loss, gravity, total=None, scale=None):
        self._init = init
        self._final = final
        self._total = total
        self._loss = loss
        self._scale = scale
        self._gravity = gravity
        self._eps = 0.001
        self._height = self._init - self._final
        if self._scale == None:
            if self._total == None:
                self._scale = 1
            else:
                n = np.log(self._eps / self._height) / np.log(self._loss)
                sqrt_l = np.sqrt(self._loss)
                self._scale = self._total / (2 * np.sqrt(2 * self._height / self._gravity) *
                                             (sqrt_l * (1 - sqrt_l ** n) / (1 - sqrt_l) + 0.5))
        self.reset()

    def reset(self):
        self._t = 0
        self._is_terminate = False
        self._v = 0
        self._height = self._init - self._final
        self._stage_t = np.sqrt(2 * self._height / self._gravity)
        self._staged_t = self._stage_t
        self._mid_t = 0
        self._h_func = lambda: self._final + self._height - \
                               self._gravity * ((self._t / self._scale - self._mid_t) ** 2) / 2

    def step(self):
        self._h = self._h_func()
        self._t += 1
        if not self._is_terminate and self._t / self._scale > self._staged_t:
            self._height *= self._loss
            if self._height < self._eps:
                self._is_terminate = True
                self._h_func = lambda: self._final
            self._stage_t = np.sqrt(2 * self._height / self._gravity)
            self._mid_t = self._staged_t + self._stage_t
            self._staged_t += 2 * self._stage_t
        return self._h


if __name__ == '__main__':
    be = BounceEpsilon(1.0, 0.01, 0.7, 0.0001, total=20000)
    heights = []
    for _ in range(20000):
        heights.append(be.step())
    print(heights)
    with open('height.csv', 'w') as f:
        [f.write('%f\n' % i) for i in heights]
