"""Opening figure: success rate plotted against number of parameters *for baselines* and new scales

x-axis: number of parameters
y-axis: success rate
colour: algorithm (mtmhsac, PaCo, MOORE, SM)
"""

import pathlib
from dataclasses import dataclass

import altair as alt
import design_system
import polars as pl
from get_data import get_metric

import tyro


@dataclass
class Args:
    benchmark: str = "MT10"


def main():
    args = tyro.cli(Args)

    entity = "reggies-phd-research"
    metric = "charts/mean_success_rate"

    def project(benchmark: str):
        if benchmark == "MT10":
            return "mtrl-mt10-results"
        elif benchmark == "MT50":
            return "mtrl-mt50-results"
        else:
            raise ValueError

    def run_name_width(benchmark: str, width: int):
        if benchmark == "MT10":
            return f"mt10_mtmhsac_v2_width_{width}"
        elif benchmark == "MT50":
            return f"mt50_mtmhsac_v2_{width}_width"
        else:
            raise ValueError

    def run_name_baseline(benchmark: str, baseline: str):
        if benchmark == "MT10":
            if baseline.lower() == "paco":
                return "mt10_mtmhsac_paco_params_v2_3_layers"
            elif baseline.lower() == "moore":
                return "mt10_mtmhsac_v2_moore_params_3_layers"
            elif baseline.lower() == "sm":
                return "mt10_mtmhsac_sm_params_3_layers_v2"
            else:
                raise ValueError
        elif benchmark == "MT50":
            if baseline.lower() == "sm":
                return "mt50_mtmhsac_sm_params_v2"
            elif baseline.lower() == "moore":
                return "mtmhsac_moore_params_log_std-10_clipped_q"
            elif baseline.lower() == "paco":
                return "mt50_mtmhsac_v2_paco_params_3_layers"
            else:
                raise ValueError
        else:
            raise ValueError

    data = pl.DataFrame(
        [
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_mtmhsac"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_mtmhsac_v2",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_mtmhsac"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_mtmhsac_v2",
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_softmodules_task_weights_false"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_softmodules_depth_4",  # HACK:
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_baseline(args.benchmark, "SM"),
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    f"{args.benchmark.lower()}_paco",  # HACK:
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_baseline(args.benchmark, "PaCo"),
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_moore_fix"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_moore",  # HACK:
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_baseline(args.benchmark, "MOORE"),
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 1024),
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 1024),
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 2048),
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 2048),
                    metric,
                ),
            },
            {
                "Algorithm": "Simple baseline",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 4096),
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    run_name_width(args.benchmark, 4096),
                    metric,
                ),
            },
            {
                "Algorithm": "Soft Modules",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_softmodules_task_weights_false"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_softmodules_depth_4",  # HACK:
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_softmodules_task_weights_false"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_softmodules_depth_4",  # HACK:
                    metric,
                ),
            },
            {
                "Algorithm": "PaCo",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    f"{args.benchmark.lower()}_paco",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    f"{args.benchmark.lower()}_paco",
                    metric,
                ),
            },
            {
                "Algorithm": "MOORE",
                "Number of parameters": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_moore_fix"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_moore",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    entity,
                    project(args.benchmark),
                    "mt10_moore_fix"
                    if args.benchmark.lower() == "mt10"
                    else "mt50_moore",
                    metric,
                ),
            },
        ]
    ).explode("Success rate")

    # IQM
    iq_data = data.group_by("Algorithm", "Number of parameters").agg(
        pl.col("Success rate").quantile(0.25).alias("q1"),
        pl.col("Success rate").quantile(0.75).alias("q3"),
    )
    data = data.join(
        iq_data, on=["Algorithm", "Number of parameters"], how="left"
    ).filter(
        (pl.col("Success rate") >= pl.col("q1"))
        & (pl.col("Success rate") <= pl.col("q3"))
    )

    x_axis = alt.X(
        "Number of parameters:Q",
        scale=alt.Scale(
            type="log",
            domain=[  # pyright: ignore [reportArgumentType]
                data["Number of parameters"].min(),
                data["Number of parameters"].max(),
            ],
        ),
        title="Number of parameters",
        axis=alt.Axis(
            format="~s",
            labelExpr="datum.value >= 1000000 ? format(datum.value / 1000000, '.0f') + 'M' : datum.value >= 1000 ? format(datum.value / 1000, '.0f') + 'K' : datum.value",
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
            values=[
                data["Number of parameters"].min(),  # pyright: ignore [reportArgumentType]
                # 200_000,
                # 500_000,
                1_000_000,
                2_000_000,
                5e6,
                # 9e6,
                10_000_000,
                30_000_000,
            ],
        ),
    )
    y_axis = alt.Y(
        "mean(Success rate):Q",
        title="Success rate",
        scale=alt.Scale(domain=[0.6 if args.benchmark.lower() == "mt10" else 0.3, 1]),
    )
    color_axis = alt.Color(
        "Algorithm:N",
        title="Algorithm",
        legend=alt.Legend(orient="top-left", symbolOpacity=1.0, symbolSize=50),
    ).scale(
        domain=["Simple baseline", "Soft Modules", "MOORE", "PaCo"],
        range=[
            design_system.COLORS["primary"][500],
            design_system.COLORS["support 2"][600],
            design_system.COLORS["support 1"][700],
            design_system.COLORS["support 2"][900],
        ],
    )

    # Create the scaling chart
    base = alt.Chart(data).encode(
        x=x_axis,
        y=y_axis,
        color=color_axis,
    )

    line = base.mark_line(point=alt.OverlayMarkDef(size=100)).encode(
        shape=alt.Shape(
            "Algorithm:N",
            title="Algorithm",
        ).scale(
            domain=["Simple baseline", "Soft Modules", "MOORE", "PaCo"],
            range=[
                "circle",
                "triangle-down",
                "square",
                "diamond",
            ],
        )
    )
    band = base.mark_errorband(extent="ci")
    baseline_errorbar = (
        alt.Chart(data.filter(pl.col("Algorithm") != "Simple baseline"))
        .encode(
            x=x_axis,
            y=y_axis,
            color=color_axis,
        )
        .mark_errorbar(extent="ci")
    )
    chart = band + line + baseline_errorbar
    chart = (
        chart.properties(
            width=600,
            height=400,
            # title=f"Opening figure {args.benchmark}"
        )
        .configure_title(
            font=design_system.PRIMARY_FONT,
            fontSize=20,
        )
        .configure_axis(
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
            labelFontSize=14,
            titleFontSize=16,
        )
        .configure_legend(
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
            labelFontSize=14,
            titleFontSize=16,
        )
    )

    # Save the chart
    figures_dir = pathlib.Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    chart.save(figures_dir / f"fig1_new_{args.benchmark.lower()}.svg")


if __name__ == "__main__":
    main()
