"""Both backbones, pretrained and from scratch, map [2, 1, 512, 512] images to the final norm's [2, 1024, 768]
patch tokens; pretrained=True really loads the published weights."""
import pytest
import torch

from sarbench.backbones import build_backbone

FINAL_NORM = {'vit': 'vit.norm', 'terramind': 'terramind.encoder_norm'}


@pytest.mark.parametrize('pretrained', [True, False])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_patch_tokens(name, pretrained):
    backbone = build_backbone(name, pretrained).cuda().eval()
    normed = []
    backbone.get_submodule(FINAL_NORM[name]).register_forward_hook(lambda module, args, out: normed.append(out))
    with torch.no_grad():
        tokens = backbone(torch.randn(2, 1, 512, 512, device='cuda'))
    assert tokens.shape == (2, 1024, 768)  # 32 x 32 patches, no CLS token
    assert tokens.isfinite().all()
    assert torch.equal(tokens, normed[0][:, -1024:])  # the final norm's output; ViT's norm also sees the CLS token
    assert len(backbone.blocks) == 12 and backbone.embed_dim == 768


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_pretrained_weights_are_loaded(name):
    first, second, scratch = (dict(build_backbone(name, p).named_parameters()) for p in (True, True, False))
    assert all(torch.equal(first[k], second[k]) for k in first)  # no weight left at a random init
    assert not any(torch.equal(first[k], scratch[k]) for k in first)  # nor at the scratch init
