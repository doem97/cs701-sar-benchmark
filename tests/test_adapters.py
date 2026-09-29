"""Adapters: exact no-ops at init, only adapter weights train, MoE-LoRA routes per task."""
import pytest
import torch
from torch import nn

from adapters import LoRA, MoELoRA, add_adapters, set_task
from backbones import build_backbone

# 12 blocks x rank 16 x ((768 + 2304) + (768 + 768)) for A and B of qkv and proj;
# MoE-LoRA: the same A and B sizes (4 experts x rank 4 = 16), plus 12 blocks x 2 layers x 2 routers x 768 x 4 experts
TRAINABLE = {'lora': 884_736, 'moelora': 1_032_192}
ADAPTER_WEIGHTS = ('.down.weight', '.up.weight', '.routers.0.weight', '.routers.1.weight')


def tokens(backbone, x):
    """Tokens without gradients on the GPU; .cuda() also moves adapters added since the last call."""
    with torch.no_grad():
        return backbone.cuda().eval()(x)


def tokens_per_task(name, kind):
    """Tokens under task 0 and task 1, after giving every adapter a non-zero B."""
    backbone = build_backbone(name, pretrained=False)
    add_adapters(backbone, kind)
    for layer in backbone.modules():
        if isinstance(layer, (LoRA, MoELoRA)):
            nn.init.normal_(layer.up.weight, std=0.02)
    x = torch.randn(2, 1, 512, 512, device='cuda')
    set_task(backbone, 0)
    task0 = tokens(backbone, x)
    set_task(backbone, 1)
    return task0, tokens(backbone, x)


@pytest.mark.parametrize('kind', ['lora', 'moelora'])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_adapters_are_noops_at_init(name, kind):
    backbone = build_backbone(name, pretrained=True)
    x = torch.randn(2, 1, 512, 512, device='cuda')
    before = tokens(backbone, x)
    add_adapters(backbone, kind)
    assert (tokens(backbone, x) - before).abs().max() <= 1e-6


@pytest.mark.parametrize('kind', ['lora', 'moelora'])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_only_adapter_weights_train(name, kind):
    backbone = build_backbone(name, pretrained=False)
    add_adapters(backbone, kind)
    trainable = {n: p.numel() for n, p in backbone.named_parameters() if p.requires_grad}
    assert all(n.endswith(ADAPTER_WEIGHTS) for n in trainable)
    assert sum(trainable.values()) == TRAINABLE[kind]


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_moelora_tokens_depend_on_task(name):
    task0, task1 = tokens_per_task(name, 'moelora')
    assert (task0 - task1).abs().max() > 1e-3


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_lora_tokens_do_not_depend_on_task(name):
    task0, task1 = tokens_per_task(name, 'lora')
    assert torch.equal(task0, task1)


def test_moelora_is_a_gated_sum_of_lora_experts():
    """The stacked down/up layers compute y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x."""
    # 3 experts of rank 2 (unequal, so mixing up the two axes fails); alpha / r = 2
    layer = MoELoRA(nn.Linear(32, 24), experts=3, rank=2, alpha=4).double()
    nn.init.normal_(layer.up.weight)
    set_task(layer, 1)
    x = torch.randn(2, 5, 32, dtype=torch.float64)  # float64: unaffected by reduced-precision fp32 matmul settings
    gates = layer.routers[1](x).softmax(dim=-1)
    A = layer.down.weight.split(2)  # A_e: [2, 32]
    B = layer.up.weight.split(2, dim=1)  # B_e: [24, 2]
    experts = sum(gates[..., e, None] * (x @ A[e].T @ B[e].T) for e in range(3))
    torch.testing.assert_close(layer(x), layer.base(x) + 2 * experts)
