from typing import Tuple

import nltk
import typer
import torch
import numpy as np
import polars as pl
from rich import print, progress
import plotly_express as px
from transformer_lens import HookedTransformer
from sae_lens.sae import SAE
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
from sae_lens.tutorial.tsea import (
    get_enrichment_df,
    manhattan_plot_enrichment_scores,
    plot_top_k_feature_projections_by_token_and_category,
    get_letter_gene_sets,
    generate_pos_sets,
    get_test_gene_sets,
    get_gene_set_from_regex,
)

app = typer.Typer()
nltk.download("averaged_perceptron_tagger")


@torch.no_grad()
def get_feature_property_df(sae: SAE, feature_sparsity: torch.Tensor) -> pl.DataFrame:
    W_dec_normalized = sae.W_dec.cpu()
    W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T
    d_e_projection = (W_dec_normalized * W_enc_normalized).sum(
        -1
    )  # measuring the alignment between detection and reconstruction
    # Feature detects "cooking words" → produces "cooking words" -> Well-behaved, interpretable feature
    # Feature detects one pattern but produces something unrelated -> Might be a "composite feature" or poorly trained
    # Feature detects a pattern but actively suppresses it in output -> Could be an "inhibitory" feature
    b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T
    # Large positive = bias strongly aligns with that feature
    # Zero = bias is perpendicular to that feature
    # Negative = bias points away from that feature

    return pl.DataFrame(
        {
            "log_feature_sparsity": feature_sparsity + 1e-10,
            "d_e_projection": d_e_projection,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec_projection": b_dec_projection,
            "feature": list(range(sae.cfg.d_sae)),  # type: ignore
            "dead_neuron": (feature_sparsity < -9).cpu(),
        }
    )


@torch.no_grad()
def get_stats_df(projection: torch.Tensor):
    """
    Returns a dataframe with the mean, std, skewness and kurtosis of the projection
    """
    mean = projection.mean(dim=1, keepdim=True)
    diffs = projection - mean
    var = (diffs**2).mean(dim=1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1)
    kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=1)

    return pl.DataFrame(
        {
            "feature": range(len(skews)),
            "mean": mean.numpy().squeeze(),
            "std": std.numpy().squeeze(),
            "skewness": skews.numpy(),
            "kurtosis": kurtosis.numpy(),
        }
    )


@torch.no_grad()
def get_all_stats_dfs(
    gpt2_small_sparse_autoencoder: dict[str, SAE],  # [hook_point, sae]
    gpt2_small_sae_sparsities: dict[str, torch.Tensor],  # [hook_point, sae]
    model: HookedTransformer,
    cosine_sim: bool = False,
):
    stats_dfs = []
    with progress.Progress() as p_bar:
        task = p_bar.add_task("layers", total=len(gpt2_small_sparse_autoencoder.keys()))
        for key, sparse_autoencoder in gpt2_small_sparse_autoencoder.items():
            layer = int(key.split(".")[1])
            p_bar.update(
                task, description="Processing layer {sparse_autoencoder.cfg.hook_name}"
            )
            W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(
                sparse_autoencoder.W_dec.cpu(), model, cosine_sim
            )
            log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
            W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
            W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
            stats_dfs.append(W_U_stats_df_dec)

    return stats_dfs


@torch.no_grad()
def get_W_U_W_dec_stats_df(
    W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False
) -> Tuple[pl.DataFrame, torch.Tensor]:
    W_U = model.W_U.detach().cpu()
    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    dec_projection_onto_W_U = W_dec @ W_U
    W_U_stats_df = get_stats_df(dec_projection_onto_W_U)
    return W_U_stats_df, dec_projection_onto_W_U


@app.command()
def main():
    # params
    layer = 8

    # loading GPT2 small and SAE weights
    model = HookedTransformer.from_pretrained("gpt2-small")
    gpt2_small_sparse_autoencoder = {}
    gpt2_small_sae_sparsities = {}

    for layer in range(12):
        sae, original_cfg_dict, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device="cpu",
        )
        gpt2_small_sparse_autoencoder[f"blocks.{layer}.hook_resid_pre"] = sae
        gpt2_small_sae_sparsities[f"blocks.{layer}.hook_resid_pre"] = sparsity

    # get the sae and sparsity for the layer
    sparse_autoencoder = gpt2_small_sparse_autoencoder[f"blocks.{layer}.hook_resid_pre"]
    log_feature_sparsity = gpt2_small_sae_sparsities[
        f"blocks.{layer}.hook_resid_pre"
    ].cpu()
    W_dec = sparse_autoencoder.W_dec.detach().cpu()
    W_U_stats_df_dec, dec_projection_onto_W_U = get_W_U_W_dec_stats_df(
        W_dec, model, cosine_sim=False
    )
    W_U_stats_df_dec = W_U_stats_df_dec.with_columns(
        pl.Series(name="sparsity", values=log_feature_sparsity)
    )
    print(W_U_stats_df_dec)

    # plot the skewness of the logit weight distributions
    # px.histogram(
    #     W_U_stats_df_dec,
    #     x="skewness",
    #     width=800,
    #     height=300,
    #     nbins=1000,
    #     title="Skewness of the Logit Weight Distributions",
    # ).show()

    # px.histogram(
    #     np.log10(W_U_stats_df_dec["kurtosis"].to_numpy()),
    #     width=800,
    #     height=300,
    #     nbins=1000,
    #     title="Kurtosis of the Logit Weight Distributions",
    # ).show()

    # # plot form the paper
    # fig = px.scatter(
    #     W_U_stats_df_dec,
    #     x="skewness",
    #     y="kurtosis",
    #     color="std",
    #     color_continuous_scale="Portland",
    #     hover_name="feature",
    #     width=800,
    #     height=500,
    #     log_y=True,
    #     labels={
    #         "x": "Skewness",
    #         "y": "Kurtosis",
    #         "color": "Standard Deviation",
    #     },
    #     title="Layer 8: Skewness vs Kurtosis of the Logit Weight Distributions",
    # )
    # fig.update_traces(marker=dict(size=3))
    # fig.show()

    # then you can query across combinations of the statistics to find features of interest and open them in neuronpedia.
    # tmp_df = W_U_stats_df_dec[["feature", "skewness", "kurtosis", "std"]]
    # tmp_df = (
    #     tmp_df.filter(pl.col("skewness") > 3.0)
    #     .sort("skewness", descending=True)
    #     .head(10)
    # )
    # get_neuronpedia_quick_list(sparse_autoencoder, tmp_df["feature"].to_list())

    # define token sets
    vocab = model.tokenizer.get_vocab()  # type: ignore
    regex_dict = {
        "starts_with_space": r"Ġ.*",
        "starts_with_capital": r"^Ġ*[A-Z].*",
        "starts_with_lower": r"^Ġ*[a-z].*",
        "all_digits": r"^Ġ*\d+$",
        "is_punctuation": r"^[^\w\s]+$",
        "contains_close_bracket": r".*\).*",
        "contains_open_bracket": r".*\(.*",
        "all_caps": r"Ġ*[A-Z]+$",
        "1 digit": r"Ġ*\d{1}$",
        "2 digits": r"Ġ*\d{2}$",
        "3 digits": r"Ġ*\d{3}$",
        "4 digits": r"Ġ*\d{4}$",
        "length_1": r"^Ġ*\w{1}$",
        "length_2": r"^Ġ*\w{2}$",
        "length_3": r"^Ġ*\w{3}$",
        "length_4": r"^Ġ*\w{4}$",
        "length_5": r"^Ġ*\w{5}$",
    }

    # print size of gene sets
    all_token_sets = get_letter_gene_sets(vocab)
    for key, value in regex_dict.items():
        gene_set = get_gene_set_from_regex(vocab, value)
        all_token_sets[key] = gene_set
    # some other sets that can be interesting
    pos_sets = generate_pos_sets(vocab)
    arbitrary_sets = get_test_gene_sets(model)
    all_token_sets = {**all_token_sets, **pos_sets}
    all_token_sets = {**all_token_sets, **arbitrary_sets}
    # for each gene set, convert to string and  print the first 5 tokens
    # for token_set_name, gene_set in sorted(
    #     all_token_sets.items(), key=lambda x: len(x[1]), reverse=True
    # ):
    #     tokens = [model.to_string(id) for id in list(gene_set)][:10]  # type: ignore
    #     print(f"{token_set_name}, has {len(gene_set)} genes")
    #     print(tokens)
    #     print("----")

    # Performing Token Set Enrichment Analysis
    # Below we perform token set enrichment analysis on various token sets. In practice, we'd likely perform tests across all tokens and large libraries of sets simultaneously but to make it easier to run, we look at features with higher skew and select of a few token sets at a time to consider.
    features_ordered_by_skew = W_U_stats_df_dec.sort("skewness", descending=True)[
        "feature"
    ].to_list()
    # filter our list.
    token_sets_index = [
        "starts_with_space",
        "starts_with_capital",
        "all_digits",
        "is_punctuation",
        "all_caps",
    ]
    token_set_selected = {
        k: set(v) for k, v in all_token_sets.items() if k in token_sets_index
    }
    df_enrichment_scores = get_enrichment_df(
        dec_projection_onto_W_U,  # use the logit weight values as our rankings over tokens.
        features_ordered_by_skew,  # subset by these features
        token_set_selected,  # use token_sets
    )
    print(df_enrichment_scores)
    manhattan_plot_enrichment_scores(
        df_enrichment_scores,
        label_threshold=0,
        top_n=3,  # use our enrichment scores
    ).show()

    fig = px.scatter(
        df_enrichment_scores.apply(lambda x: -1 * np.log(1 - x)).T,
        x="starts_with_space",
        y="starts_with_capital",
        marginal_x="histogram",
        marginal_y="histogram",
        labels={
            "starts_with_space": "Starts with Space",
            "starts_with_capital": "Starts with Capital",
        },
        title="Enrichment Scores for Starts with Space vs Starts with Capital",
        height=800,
        width=800,
    )
    # reduce point size on the scatter only
    fig.update_traces(marker=dict(size=2), selector=dict(mode="markers"))
    fig.show()


if __name__ == "__main__":
    nltk.download("averaged_perceptron_tagger")
    app()
