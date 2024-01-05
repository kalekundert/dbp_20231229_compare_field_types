#!/usr/bin/env python3

"""\
Compare different equivariant field types.

Usage:
    compare_field_types.py [<hparams>] [-d]

Arguments:
    <hparams>
        The name of the hyperparameters to use.  If not specified, print out a
        list of all the valid hyperparameter names.

Options:
    -d --debug
        If true, run only 10 steps and don't save any results.
"""

import torch
import torch.nn.functional as F
import sys

from atom3d_menagerie.predict import RegressionModule, get_trainer
from atom3d_menagerie.hparams import (
        label_hparams, make_hparams, require_hparams,
)
from atom3d_menagerie.data.smp import (
        VoxelizedSmpDataModule, get_default_smp_data_hparams,
)
from atom3d_menagerie.models.escnn import (
        EquivariantCnn,
        conv_bn_fourier,
        conv_bn_gated,
        pool_conv,
        invariant_fourier,
        linear_relu_dropout,
        linear_bn_relu,
)
from atompaint.datasets.voxelize import ImageParams, Grid
from atompaint.encoders.layers import (
        make_trivial_field_type,
        make_fourier_field_types,
        make_polynomial_field_types as _make_polynomial_field_types,
)
from atompaint.utils import get_scalar
from escnn.gspaces import rot3dOnR3
from escnn.nn import FieldType
from torch.optim import Adam
from dataclasses import dataclass
from more_itertools import zip_broadcast, always_iterable, flatten
from functools import partial
from pathlib import Path

info = partial(print, "INFO:", file=sys.stderr)

# Pool with strided conv
# Default to GELU, but also use gated as necessary

# - Channels will be hard-coded to be roughly constant
# - First layer will always be trivial, last layer will always be quotient.
# - Three layers in between

# Fourier  field type has unpack argument, that we can use with gated 
# nonlinearity.  Polynomial doesn't really have same concept.

# unpack: option if Fourier *and* gated.

@dataclass
class Fourier:
    quotient: bool
    max_freqs: int

    def __str__(self):
        name = "quotient" if self.quotient else "regular"
        max_freqs = '_'.join(str(x) for x in always_iterable(self.max_freqs))
        return f'{name}_max_freqs_{max_freqs}_fourier_gelu'

    def make_field_types_factory(self):
        if self.quotient:
            factory = make_quotient_field_types
        else:
            factory = make_regular_field_types

        return partial(
                factory,
                max_freqs=self.max_freqs,
        )

    def make_conv_layer(self, gspace):
        return make_fourier_layer(gspace)

@dataclass
class FourierGated(Fourier):
    unpack: bool

    def __str__(self):
        unpack = '_unpack' if self.unpack else ''
        return super().__str__().replace('fourier', 'gated') + unpack

    def make_field_types_factory(self):
        return partial(
                super().make_field_types_factory(),
                unpack=self.unpack,
        )

    def make_conv_layer(self, gspace):
        return make_gated_layer(gspace)

@dataclass
class PolynomialGated:
    terms: int

    def __str__(self):
        terms = '_'.join(str(x) for x in always_iterable(self.terms))
        return f'polynomial_terms_{terms}_gated_gelu'
    
    def make_field_types_factory(self):
        return partial(
                make_polynomial_field_types,
                terms=self.terms,
        )

    def make_conv_layer(self, gspace):
        return make_gated_layer(gspace)

@dataclass
class SingleIrrepGated:
    irreps: int

    def __str__(self):
        irreps = '_'.join(str(x) for x in always_iterable(self.irreps))
        return f'single_irrep_{irreps}_gated_gelu'
    
    def make_field_types_factory(self):
        return partial(
                make_single_irrep_field_types,
                irreps=self.irreps,
        )

    def make_conv_layer(self, gspace):
        return make_gated_layer(gspace)

HPARAMS = label_hparams(
        str,
        *make_hparams(
            Fourier,
            quotient=[False],
            max_freqs=[1, 2, [1, 2, 2]],
        ),
        *make_hparams(
            Fourier,
            quotient=[True],
            max_freqs=[1, 2, 3, 4, 5, [4, 5, 5]],
        ),
        *make_hparams(
            FourierGated,
            quotient=[False],
            max_freqs=[1, 2, [1, 2, 2]],
            unpack=[True, False],
        ),
        *make_hparams(
            FourierGated,
            quotient=[True],
            max_freqs=[1, 2, 3, 4, 5, [4, 5, 5]],
            unpack=[True, False],
        ),
        *make_hparams(
            PolynomialGated,
            terms=[1, 2, 3, 4, [3, 4, 4]],
        ),
        *make_hparams(
            SingleIrrepGated,
            irreps=[1, 2, 3],
        ),
)

def make_escnn_model(hparams):
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    grid_s2 = so3.sphere_grid('thomson_cube', N=4)

    latent_field_types = list(
            hparams.make_field_types_factory()(
                gspace,
                channels=[32, 64, 128],
            )
    )
    L = max(flatten(latent_field_types[-1].irreps))

    return EquivariantCnn(
            field_types=[
                make_trivial_field_type(gspace, 5),
                *latent_field_types,
                *make_quotient_field_types(
                    gspace,
                    channels=[256],
                    max_freqs=L,
                ),
            ],
            conv_factory=hparams.make_conv_layer(gspace),
            pool_factory=partial(
                pool_conv,
                padding=0,
            ),
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_fourier,
                ift_grid=grid_s2,
                function=F.gelu,
            ),
            mlp_channels=[512],
            mlp_factory=partial(
                linear_relu_dropout,
                drop_rate=0.25,
            ),
    )

def make_data():
    data_hparams = get_default_smp_data_hparams()
    data_hparams['img_params'] = ImageParams(
        grid=Grid(
            length_voxels=19,
            resolution_A=1.0,
        ),
        channels=['H', 'C', 'O', 'N', 'F'],
        element_radii_A=1.0,
    )
    return VoxelizedSmpDataModule(**data_hparams)

def check_invariance(model):
    from atompaint.vendored.escnn_nn_testing import (
            check_invariance, get_exact_3d_rotations,
    )

    in_type = model.layers[0].in_type
    so3 = in_type.gspace.fibergroup

    model.eval()
    check_invariance(
            model,
            in_type=in_type,
            in_tensor=torch.randn(2, 5, 19, 19, 19),
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )
    model.train()

    info(model)
    info("confirmed model invariance to exact 3D rotations")


def make_regular_field_types(gspace, channels, max_freqs, unpack=False):
    rho_widths = {
            1: 1 + 9,
            2: 1 + 9 + 25,
    }
    rho_multiplicities = get_rho_multiplicities(
            channels,
            get_value_or_values(rho_widths, max_freqs),
    )

    info("making regular Fourier representations")
    info("max_freqs:", max_freqs)
    info("unpack:", unpack)
    info("multiplicities:", rho_multiplicities)

    return make_fourier_field_types(
            gspace,
            channels=rho_multiplicities,
            max_frequencies=max_freqs,
            unpack=unpack,
    )

def make_quotient_field_types(gspace, channels, max_freqs, unpack=False):
    rho_widths = {
            0: 1,
            1: 1 + 3,
            2: 1 + 3 + 5,
            3: 1 + 3 + 5 + 7,
            4: 1 + 3 + 5 + 7 + 9,
            5: 1 + 3 + 5 + 7 + 9 + 11,
    }
    rho_multiplicities = get_rho_multiplicities(
            channels,
            get_value_or_values(rho_widths, max_freqs),
    )
    so2_z = False, -1

    info("making quotient Fourier representations")
    info("max_freqs:", max_freqs)
    info("unpack:", unpack)
    info("multiplicities:", rho_multiplicities)

    return make_fourier_field_types(
            gspace,
            channels=rho_multiplicities,
            max_frequencies=max_freqs,
            subgroup_id=so2_z,
            unpack=unpack,
    )

def make_polynomial_field_types(gspace, channels, terms):
    rho_widths = {
            1: 1,
            2: 1 + 3,
            3: 1 + 3 + 9,
            4: 1 + 3 + 9 + 27,
    }
    rho_multiplicities = get_rho_multiplicities(
            channels,
            get_value_or_values(rho_widths, terms),
    )

    info("making polynomial representations")
    info("terms:", terms)
    info("multiplicities:", rho_multiplicities)

    return _make_polynomial_field_types(
            gspace,
            channels=rho_multiplicities,
            terms=terms,
    )

def make_single_irrep_field_types(gspace, channels, irreps):
    # I have a feeling that this won't work at all.  Since the input irreps are 
    # all trivial, I don't think any information can get from the inputs to the 
    # hidden layers.  But this Schur's lemma stuff is a part of ESCNN that I 
    # don't understand very well, so it's worth trying anyways.

    rho_widths = {
            0: 1,
            1: 3,
            2: 5,
            3: 7,
            4: 9,
            5: 11,
    }
    rho_multiplicities = get_rho_multiplicities(
            channels,
            get_value_or_values(rho_widths, irreps),
    )

    info("making single-irrep representations")
    info("irreps:", irreps)
    info("multiplicities:", rho_multiplicities)

    for i, multiplicity_i in enumerate(rho_multiplicities):
        irrep_i = get_scalar(irreps, i)
        assert irrep_i >= 0

        yield FieldType(
                gspace,
                multiplicity_i * [gspace.fibergroup.irrep(irrep_i)],
        )


def make_fourier_layer(gspace):
    so3 = gspace.fibergroup
    grid = so3.grid('thomson_cube', N=4)
    return partial(
            conv_bn_fourier,
            ift_grid=grid,
            function=F.gelu,
    )

def make_gated_layer(gspace):
    return partial(
            conv_bn_gated,
            function=F.gelu,
    )

def get_rho_multiplicities(channels, rho_widths):
    return [
            round(c / w)
            for c, w in zip_broadcast(channels, rho_widths, strict=True)
    ]

def get_value_or_values(map, key_or_keys):
    try:
        return map[key_or_keys]
    except (KeyError, TypeError):
        return [map[k] for k in key_or_keys]


if __name__ == '__main__':
    import docopt

    args = docopt.docopt(__doc__)
    hparams_name, hparams = require_hparams(args['<hparams>'], HPARAMS)

    model = make_escnn_model(hparams)
    data = make_data()

    check_invariance(model)

    trainer = get_trainer(
            Path(hparams_name),
            max_epochs=50,
            fast_dev_run=args['--debug'] and 10,
    )
    model = RegressionModule(model, Adam)
    trainer.fit(model, data)
