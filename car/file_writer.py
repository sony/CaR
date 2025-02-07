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
import pathlib

import numpy as np


class FileWriter:
    def __init__(self, outdir, file_prefix, fmt="%.3f"):
        super(FileWriter, self).__init__()
        if isinstance(outdir, str):
            outdir = pathlib.Path(outdir)
        self._outdir = outdir
        if not self._outdir.exists():
            self._outdir.mkdir(parents=True)
        self._file_prefix = file_prefix
        self._fmt = fmt

    def write_scalar(self, iteration_num, scalar):
        outfile = self._outdir / (self._file_prefix + "_scalar.tsv")

        len_scalar = len(scalar.values())
        out_scalar = {}
        out_scalar["iteration"] = iteration_num
        out_scalar.update(scalar)

        self._create_file_if_not_exists(outfile, out_scalar.keys())

        with open(outfile, "a") as f:
            np.savetxt(f, [list(out_scalar.values())], fmt=["%i"] + [self._fmt] * len_scalar, delimiter="\t")

    def _create_file_if_not_exists(self, outfile, header_keys):
        if not outfile.exists():
            outfile.touch()
            self._write_file_header(outfile, header_keys)

    def _write_file_header(self, filepath, keys):
        with open(filepath, "w+") as f:
            np.savetxt(f, [list(keys)], fmt="%s", delimiter="\t")
