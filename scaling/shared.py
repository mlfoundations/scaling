import json
import os
import scipy
import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D

from scaling.constants import *
from scaling.laws import *


def parse_model_json(model_json, cc_mults, datasets, eval_dir=None):
    payload = {}
    data = None
    with open(model_json) as f:
        data = json.load(f)

    payload["cc_mult"] = data["hyperparameters"]["chinchilla_multiplier"]
    payload["dataset_name"] = data["dataset_name"]

    if payload["cc_mult"] not in cc_mults:
        return None
    if payload["dataset_name"] not in datasets:
        return None

    payload["name"] = data["name"]
    payload["model_name"] = data["hyperparameters"]["model"].split("/")[-1].split(".")[0]
    payload["N"] = data["hyperparameters"]["params"]
    payload["D"] = data["hyperparameters"]["tokens"]
    payload["flops"] = 6 * payload["N"] * payload["D"]
    payload["color"] = DATASET_COLORS[payload["dataset_name"]]
    payload["shape"] = MODEL_SHAPES[payload["model_name"]] if payload["model_name"] in MODEL_SHAPES else "o"
    payload["tok_mult"] = payload["cc_mult"] * 20

    for result in data["results"]:
        suffix = result["val_data"][0].split("/")[-2]
        if "de-en" in suffix:
            suffix = result["val_data"][0].split("/")[-1].split(".")[0]
        payload[f"loss_{suffix}"] = result["loss"]
        payload[f"loss_upper_{suffix}"] = result["loss_sequences_upper_95"]
        payload[f"loss_lower_{suffix}"] = result["loss_sequences_lower_95"]

    if eval_dir is not None:

        root_name = f"evaluation_{Path(model_json).stem}_heavy.json"
        eval_json = f"{eval_dir}/{root_name}"
        assert os.path.exists(eval_json)

        eval_data = None
        with open(eval_json) as f:
            eval_data = json.load(f)

        err_acc = 0.0
        err_subset_acc = 0.0

        err_acc_count = 0
        err_subset_acc_count = 0

        for k in eval_data["eval_metrics"]["icl"]:
            err = 1.0 - eval_data["eval_metrics"]["icl"][k]
            if k in FRIENDLY_CITATIONS:
                if k in SUBSET:
                    err_subset_acc += err
                    err_subset_acc_count += 1

                err_acc += err
                err_acc_count += 1

                payload[f"err_{k}"] = err

        assert err_subset_acc_count == len(SUBSET)

        payload["err_avg"] = err_acc / err_acc_count
        payload["err_avg_subset"] = err_subset_acc / err_subset_acc_count

    return payload


def parse_model_jsons(
    model_dir,
    datasets,
    cc_mults=[
        1.0,
    ],
    eval_dir=None,
):
    payloads = []
    for fp in os.listdir(model_dir):
        parsed = parse_model_json(f"{model_dir}/{fp}", cc_mults, datasets, eval_dir)
        if parsed is not None:
            payloads.append(parsed)

    df = pd.DataFrame.from_dict(payloads).sort_values(by=["flops"])

    return df


def split_df_by_dataset(df):
    names = df["dataset_name"].unique().tolist()  # find unique values
    dfs = []
    for n in names:
        dfs.append(df[df["dataset_name"] == n].sort_values(by=["flops"]).reset_index(drop=True))

    return dfs


def split_df_by_mult(df, included_models):

    dff = df[df["model_name"].isin(included_models)]
    names = dff["cc_mult"].unique().tolist()  # find unique values
    dfs = []
    for n in names:
        dfs.append(dff[dff["cc_mult"] == n].sort_values(by=["flops"]).reset_index(drop=True))

    return dfs, names


def split_df_by_model(df, min_only_field=""):
    names = df["model_name"].unique().tolist()  # find unique values
    dfs = []
    for n in names:
        df_tmp = df[df["model_name"] == n].sort_values(by=["dataset_name", "cc_mult"])

        if len(min_only_field):
            min_inds = df_tmp[min_only_field].idxmin()
            df_tmp = df_tmp.loc[[min_inds]]

        dfs.append(df_tmp.reset_index(drop=True))

    return dfs, names


def fit_ds(train_dataset, val_dataset, downstream, model_dir, eval_dir, Ms, add_1b, return_points):
    df = parse_model_jsons(
        model_dir,
        cc_mults=Ms,
        datasets=[train_dataset],
        eval_dir=eval_dir,
    )

    points_loss = None
    points_error = None

    fit_models = ["d=96_l=8_h=4", "d=512_l=8_h=4", "d=576_l=24_h=8", "d=1024_l=24_h=8"]
    df_mults, names = split_df_by_mult(df, fit_models)

    df_mults_dict = {names[i]: df_mults[i] for i in range(len(names))}

    xs_irr = df_mults_dict[1.0]["flops"].tolist()
    ys_irr = df_mults_dict[1.0][f"loss_{val_dataset}"].tolist()
    ys2_irr = df_mults_dict[1.0][f"err_{downstream}"].tolist()
    ms_irr = df_mults_dict[1.0]["tok_mult"].tolist()
    ns_irr = df_mults_dict[1.0]["N"].tolist()
    shapes_irr = df_mults_dict[1.0]["shape"].tolist()

    tmp = df_mults_dict[16.0]
    xs_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"]["flops"].tolist())
    ys_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"][f"loss_{val_dataset}"].tolist())
    ys2_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"][f"err_{downstream}"].tolist())
    ms_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"]["tok_mult"].tolist())
    ns_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"]["N"].tolist())
    shapes_irr.extend(tmp[tmp["model_name"] == "d=96_l=8_h=4"]["shape"].tolist())

    assert len(xs_irr) == 5

    if return_points:
        points_loss = {
            "flops": xs_irr.copy(),
            "loss": ys_irr.copy(),
            "error": ys2_irr.copy(),
            "mults": ms_irr.copy(),
            "params": ns_irr.copy(),
            "shapes": shapes_irr.copy(),
        }

    popt_ours = curve_fit_powlaw_ours(np.array([ns_irr, ms_irr]), np.array(ys_irr).astype(float))

    if add_1b:
        # add 1b for this
        df_ds = parse_model_jsons(model_dir, cc_mults=[1.0], datasets=[train_dataset], eval_dir=eval_dir)
        df_mults_ds, _ = split_df_by_mult(df_ds, ["open_lm_1b"])
        for ii, df_mult in enumerate(df_mults_ds):
            tmp2 = df_mult[(df_mult["model_name"] == "open_lm_1b")]
            assert len(tmp2["flops"].tolist()) == 1

            xs_irr.extend(tmp2["flops"].tolist())
            ys_irr.extend(tmp2[f"loss_{val_dataset}"].tolist())
            ms_irr.extend(tmp2["tok_mult"].tolist())
            ns_irr.extend(tmp2["N"].tolist())
            ys2_irr.extend(tmp2[f"err_{downstream}"].tolist())
            shapes_irr.extend(tmp2[f"shape"].tolist())

        assert len(xs_irr) == 6

    if return_points:
        points_error = {
            "flops": xs_irr.copy(),
            "loss": ys_irr.copy(),
            "error": ys2_irr.copy(),
            "mults": ms_irr.copy(),
            "params": ns_irr.copy(),
            "shapes": shapes_irr.copy(),
        }

    popt_ds, _ = scipy.optimize.curve_fit(
        lambda t, a, b, E: E - a * np.exp(-b * t),
        ys_irr,
        ys2_irr,
        maxfev=10000,
    )

    return (popt_ours, popt_ds), (points_loss, points_error)


def legend_helper(ax, tups):
    # create manual symbols for legend
    handles = []

    for tup in tups:
        label, color, marker, linestyle, fillstyle = tup

        handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                fillstyle=fillstyle,
            )
        )

    ax.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
    ).set_zorder(102)
