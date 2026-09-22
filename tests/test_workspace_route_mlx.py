import numpy as np
import pytest

from test_workspace_mlx import model  # noqa: F401
from voynich.workspace.route_backend import RouteWorkspace, positional_field


@pytest.mark.parametrize('layer',[0,2])
@pytest.mark.parametrize('scope',['last','earlier','all'])
def test_matrix_route_matches_individual_position_edits(model,layer,scope):  # noqa: F811
    route=RouteWorkspace.__new__(RouteWorkspace)
    route.__dict__.update(model.__dict__)
    ids=[2,3,5,7,11,13,17]
    field=positional_field(np.random.default_rng(10).normal(size=(7,16))*.1,scope)
    logits,cache,captures=route.prefill_field(ids,fields={layer:field},capture_layers=(layer,))
    patches=[{'layer':layer,'position':i,'delta':d} for i,d in enumerate(field)]
    for step in range(3):
        reference,_=model.forward(ids,patches=patches)
        np.testing.assert_allclose(logits,reference[0],atol=1e-5,rtol=1e-4)
        if step==0:
            assert captures[layer].shape==(7,16)
        ids.append(20+step)
        logits=route.cached_step(20+step,cache)


def test_complete_donor_field_reconstructs_donor_first_logits(model):  # noqa: F811
    route=RouteWorkspace.__new__(RouteWorkspace)
    route.__dict__.update(model.__dict__)
    own,donor=[1,2,3,5,7],[1,4,8,5,7]
    _,_,h=route.prefill_field(own,capture_layers=(1,))
    wanted,_,other=route.prefill_field(donor,capture_layers=(1,))
    actual,_,_=route.prefill_field(own,fields={1:other[1]-h[1]})
    np.testing.assert_allclose(actual,wanted,atol=1e-5)
