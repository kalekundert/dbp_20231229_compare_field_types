#!/usr/bin/env python3

"""
Usage:
    plot.py [-k <regex>] [-i <irrep>] [-a]

Options:
    -k --select <regex>
        Only show traces that match the given pattern.

    -i --irrep <num>
        Only show traces with the given number of irreps.

    -a --all
        Show all training curves, including those that are bad enough force the 
        axes to zoom out significantly.
"""

import polars as pl
import matplotlib.pyplot as plt
import docopt
import re

from atompaint.analysis.plot_metrics import (
        load_tensorboard_logs, plot_training_metrics,
        extract_hparams, pick_metrics,
)
from pathlib import Path
from functools import cache
from dataclasses import dataclass, fields

@cache
def parse_hparams_name(name):
    m = re.match(
            r'''(?x)
                (?P<rho>regular|quotient|polynomial|single_irrep)_
                (
                    max_freqs_(?P<max_freqs>[0-9_]+)|
                    terms_(?P<terms>[0-9_]+)|
                    (?P<irrep>[0-9_]+)
                )_
                (?P<nonlinearity>fourier_gelu|gated_gelu)
                (?P<unpack>_unpack)?
            ''',
            name,
    )

    rho = m.group('rho')

    if max_freqs := m.group('max_freqs'):
        max_irrep = max(map(int, max_freqs.split('_')))
    
    if terms := m.group('terms'):
        max_irrep = max(map(int, terms.split('_'))) - 1

    if irreps := m.group('irrep'):
        max_irrep = max(map(int, irreps.split('_')))

    nonlinearity = m.group('nonlinearity')

    if (rho in ['regular', 'quotient']) and nonlinearity.startswith('gated'):
        unpack = bool(m.group('unpack'))
    else:
        unpack = None

    return dict(
            rho=rho,
            max_irrep=max_irrep,
            nonlinearity=nonlinearity,
            unpack=unpack,
    )


if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    df = load_tensorboard_logs([Path(__file__).parent])

    df = (df
          .with_columns(
              hparams=pl.col('model').map_elements(parse_hparams_name)
          )
          .unnest('hparams')
    )

    if args['--select']:
          df = df.filter(
              pl.col('model').str.contains(args['--select']),
          )
    if args['--irrep']:
          df = df.filter(
                  max_irrep=int(args['--irrep']),
          )
    if not args['--all']:
          df = df.filter(
                  ~pl.col('model').str.contains('single_irrep_3_gated_gelu'),
          )

    metrics = pick_metrics(df, include_train=True)
    hparams = 'rho', 'max_irrep', 'nonlinearity', 'unpack'

    plot_training_metrics(df, metrics, hparams)
    plt.savefig('compare_field_types.svg')
    plt.show()
