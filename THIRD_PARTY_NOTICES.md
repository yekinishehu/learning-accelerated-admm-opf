# Third-party notices

## IEEE 118-bus case data

The files `code/case118_bus.m` and `code/case118_gen.m` contain data adapted from the MATPOWER `case118.m` data file. Their headers record conversion from the IEEE Common Data Format file `ieee118cdf.txt`, University of Washington archive, 25 August 1993, and the MATPOWER transformer correction dated 15 February 2019.

Upstream project:

- MATPOWER: https://github.com/MATPOWER/matpower
- Case-data source identified in the file headers: MATPOWER `data/case118.m`

The upstream MATPOWER license notice explicitly distinguishes software from case files:

> The code in MATPOWER is distributed under the 3-clause BSD license below. The MATPOWER case files distributed with MATPOWER are not covered by the BSD license. In most cases, the data has either been included with permission or has been converted from data available from a public source.

Accordingly, the BSD notice in `licenses/MATPOWER-LICENSE.txt` is preserved for attribution, but it is **not** represented here as a license for the IEEE 118-bus case data. The repository's MIT and CC BY licenses do not claim to relicense upstream third-party material.

## Python dependencies

The execution and analysis code uses NumPy, PyTorch, and Matplotlib. Those packages retain their respective upstream licenses; they are not bundled as source code in this repository.
