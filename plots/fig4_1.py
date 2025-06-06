"""Figure: dormant neuron ratio throughout training for one scale, for MT10/MT25/MT50

x-axis: timestep
y-axis: dormant neuron ratio
colour: benchmark
"""

import pathlib
from dataclasses import dataclass

import altair as alt
import design_system
import polars as pl
import tyro
from get_data import get_metric, get_metric_history


@dataclass
class Args:
    width: int = 2048


def main():
    args = tyro.cli(Args)

    entity = "reggies-phd-research"
    metric = "metrics/dormant_neurons_critic_0_total_ratio"

    def project(benchmark: str, width: int):
        if benchmark == "MT10":
            return "mtrl-mt10-results"
        elif benchmark == "MT50":
            return "mtrl-mt50-results"
        elif benchmark == "MT25":
            if width != 4096:
                return "mtrl-mt25-results"
            else:
                raise NotImplementedError
        else:
            raise ValueError

    def run_name(benchmark: str, width: int):
        if benchmark == "MT10":
            return f"mt10_mtmhsac_v2_width_{width}"
        elif benchmark == "MT50":
            return f"mt50_mtmhsac_v2_{width}_width"
        elif benchmark == "MT25":
            return f"mt25_mtmhsac_v2_{width}_width"
        else:
            raise ValueError

    raw_data = [
        {
            "Benchmark": benchmark,
            "Width": args.width,
            "Number of parameters": get_metric(
                entity,
                project(benchmark, args.width),
                run_name(benchmark, args.width),
                "actor_num_params",
                source="config",
            )[0],
            "Dormant neuron ratio": get_metric_history(
                entity,
                project(benchmark, args.width),
                run_name(benchmark, args.width),
                metric,
            ),
        }
        for benchmark in ["MT10", "MT50", "MT25"]
    ]

    # Expand history data into individual rows
    data = []
    for datum in raw_data:
        for i, run_data in enumerate(datum["Dormant neuron ratio"]):
            for step, value in run_data.items():
                data.append(
                    {
                        "Benchmark": datum["Benchmark"],
                        "Width": datum["Width"],
                        "Number of parameters": datum["Number of parameters"],
                        "Dormant neuron ratio": value,
                        "Number of environment steps": step,
                        "Run": i,
                    }
                )
    data = pl.DataFrame(data)

    max_timestep = 1e8
    x_axis = alt.X(
        "Number of environment steps:Q",
        scale=alt.Scale(domain=[0, max_timestep]),
        title="Number of environment steps",
        axis=alt.Axis(
            format="~s",
            labelExpr="datum.value >= 1000000 ? format(datum.value / 1000000, '.0f') + 'M' : datum.value >= 1000 ? format(datum.value / 1000, '.0f') + 'K' : datum.value",
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
        ),
    )
    y_axis = alt.Y(
        "mean(Dormant neuron ratio):Q",
        title="Dormant neuron ratio (%)",
        scale=alt.Scale(domain=[0, 50]),
    )
    color_axis = alt.Color(
        "Benchmark:N",
        title="Benchmark",
        legend=alt.Legend(orient="top-left", symbolOpacity=1.0, symbolSize=50),
    ).scale(
        domain=["MT10", "MT25", "MT50"],
        range=[
            design_system.COLORS["primary"][500],
            design_system.COLORS["primary"][800],
            design_system.COLORS["grey"][800],
        ],
    )

    base = alt.Chart(data).encode(
        x=x_axis,
        y=y_axis,
        color=color_axis,
    )

    line = base.mark_line(clip=True, interpolate="basis-open")
    band = base.mark_errorband(clip=True, extent="ci", interpolate="basis-open")
    chart = band + line
    chart = (
        chart.properties(
            width=600,
            height=400,
            # title="Dormant neuron ratio throughout training across different benchmarks",
        )
        .configure_title(
            font=design_system.PRIMARY_FONT,
            fontSize=design_system.FONT_SIZE_TOKENS["title"],
        )
        .configure_axis(
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
            labelFontSize=design_system.FONT_SIZE_TOKENS["axis_label"],
            titleFontSize=design_system.FONT_SIZE_TOKENS["axis_title"],
        )
        .configure_legend(
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
            labelFontSize=design_system.FONT_SIZE_TOKENS["legend_label"],
            titleFontSize=design_system.FONT_SIZE_TOKENS["legend_title"],
        )
    )

    figures_dir = pathlib.Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    chart.save(figures_dir / f"fig4_1_{args.width}.svg")


if __name__ == "__main__":
    main()
