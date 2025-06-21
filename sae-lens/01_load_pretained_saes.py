from functools import partial

import typer
import torch
import plotly.express as px
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate, test_prompt
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list


app = typer.Typer()
torch.set_grad_enabled(False)
device = torch.device("cuda")


@app.command()
def main():
    # load transformer model + sae
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
        device="cuda",
    )
    print(sae.cfg)

    # load dataset
    dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=sae.cfg.context_size,
        add_bos_token=sae.cfg.prepend_bos,
    )

    # l0 test and reconstruction test
    sae.eval()
    with torch.no_grad():
        batch_tokens = token_dataset[:32]["tokens"]
        _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
        feature_acts = sae.encode(cache[sae.cfg.hook_name])
        sae_out = sae.decode(feature_acts)
        del cache

        l0 = (feature_acts[:, 1:] > 0).float().sum(-1).detach()  # ignore the bos token
        print("average l0", l0.mean().item())
        px.histogram(l0.flatten().cpu().numpy()).show()

    # now let's see the cross-entropy loss with the sae reconstruction drop in
    def reconstruct_hook(activation, hook, sae_out):  # type: ignore
        return sae_out

    def zero_abl_hook(activation, hook):  # type: ignore
        return torch.zeros_like(activation)

    print("Original loss", model(batch_tokens, return_type="loss").item())
    print(
        "reconstruct loss",
        model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    sae.cfg.hook_name,
                    partial(reconstruct_hook, sae_out=sae_out),
                )
            ],
            return_type="loss",
        ).item(),
    )
    print(
        "Zero",
        model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
        ).item(),
    )

    # specific capability test
    example_prompt = "When John and Mary went to the shops, John gave the bag to"
    example_answer = " Mary"
    test_prompt(example_prompt, example_answer, model, prepend_bos=True)
    logits, cache = model.run_with_cache(example_prompt, prepend_bos=True)
    tokens = model.to_tokens(example_prompt)
    sae_out = sae.decode(sae.encode(cache[sae.cfg.hook_name]))

    def reconstruct_hook(activations, hook, sae_out):
        return sae_out

    def zero_abl_hook(mlp_out, hook):
        return torch.zeros_like(mlp_out)

    hook_name = sae.cfg.hook_name
    print("Orig", model(tokens, return_type="loss").item())
    print(
        "reconstruct",
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (
                    hook_name,
                    partial(reconstruct_hook, sae_out=sae_out),
                )
            ],
            return_type="loss",
        ).item(),
    )
    print(
        "Zero",
        model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(hook_name, zero_abl_hook)],
        ).item(),
    )
    with model.hooks(
        fwd_hooks=[
            (
                hook_name,
                partial(reconstruct_hook, sae_out=sae_out),
            )
        ]
    ):
        test_prompt(example_prompt, example_answer, model, prepend_bos=True)

    # visualize the sae features
    test_feature_idx_gpt = list(range(10)) + [14057]
    feature_vis_config_gpt = SaeVisConfig(
        hook_point=hook_name,
        features=test_feature_idx_gpt,
        minibatch_size_features=64,
        minibatch_size_tokens=256,
        verbose=True,
        device="cuda",
    )
    visualization_data_gpt = SaeVisRunner(
        feature_vis_config_gpt
    ).run(
        encoder=sae,  # type: ignore
        model=model,
        tokens=token_dataset[:10000]["tokens"],  # type: ignore
    )
    filename = "demo_feature_dashboards.html"
    save_feature_centric_vis(sae_vis_data=visualization_data_gpt, filename=filename)

    # connect with Neuronpedia: so cool!
    neuronpedia_quick_list = get_neuronpedia_quick_list(sae, test_feature_idx_gpt)
    print(neuronpedia_quick_list)


if __name__ == "__main__":
    app()
