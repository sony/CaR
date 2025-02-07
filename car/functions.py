# MIT License
#
# Copyright 2025 Sony Group Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn.functional as F


def quantile_huber_loss(x0, x1, kappa, tau):
    with torch.no_grad():
        u = x0 - x1
        # delta(u < 0)
        delta = torch.less(u, 0.0).float()
        assert delta.shape == u.shape
    if kappa <= 0.0:
        return u * (tau - delta)
    else:
        Lk = F.huber_loss(x0, x1, delta=kappa, reduction="none")
        assert Lk.shape == u.shape
        return torch.abs(tau - delta) * Lk / kappa


def _test_quantile_huber_loss():
    import numpy as np

    def huber_loss(x0, x1, kappa):
        diff = x0 - x1
        flag = (np.abs(diff) < kappa).astype(np.float32)
        return (flag) * 0.5 * (diff**2.0) + (1.0 - flag) * kappa * (np.abs(diff) - 0.5 * kappa)

    def np_quantile_huber_loss(x0, x1, kappa, tau):
        u = x0 - x1
        delta = np.less(u, np.zeros(shape=u.shape, dtype=np.float32))
        if kappa == 0.0:
            return (tau - delta) * u
        else:
            Lk = huber_loss(x0, x1, kappa=kappa)
            return np.abs(tau - delta) * Lk / kappa

    N = 10
    batch_size = 1
    x0 = np.random.normal(size=(batch_size, N))
    x1 = np.random.normal(size=(batch_size, N))
    for kappa in [0.0, 1.0]:
        tau = np.array([i / N for i in range(1, N + 1)]).reshape((1, -1))
        tau = np.repeat(tau, axis=0, repeats=batch_size)
        tau_var = torch.FloatTensor(tau)
        x0_var = torch.FloatTensor(x0)
        x1_var = torch.FloatTensor(x1)
        loss = quantile_huber_loss(x0=x0_var, x1=x1_var, kappa=kappa, tau=tau_var)

        actual = loss.detach().cpu().numpy()
        expected = np_quantile_huber_loss(x0=x0, x1=x1, kappa=kappa, tau=tau)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected, atol=1e-7)


if __name__ == "__main__":
    _test_quantile_huber_loss()
