"""Figure: dormant neuron ratio at the end of training for one scale, for MT10/MT25/MT50

x-axis: num tasks
y-axis: dormant neuron ratio
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
            "Number of tasks": num_tasks,
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
        for (benchmark, num_tasks) in [("MT10", 10), ("MT50", 50), ("MT25", 25)]
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
                        "Number of tasks": datum["Number of tasks"],
                        "Number of parameters": datum["Number of parameters"],
                        "Dormant neuron ratio": value,
                        "Timestep": step,
                        "Run": i,
                    }
                )
    data = pl.DataFrame(data)

    data = data.with_columns(pl.col("Timestep").cast(pl.Int64))
    data = data.filter(pl.col("Timestep") == pl.col("Timestep").max().over("Number of tasks"))

    x_axis = alt.X(
        "Number of tasks:O",
        title="Number of tasks",
        axis=alt.Axis(
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
        ),
    )
    y_axis = alt.Y(
        "mean(Dormant neuron ratio):Q",
        title="Dormant neuron ratio (%)",
        scale=alt.Scale(domain=[0, 35]),
    )

    base = alt.Chart(data).encode(
        x=x_axis,
        y=y_axis,
    )

    line = base.mark_line().encode(
        x=x_axis,
        y=y_axis,
        color=alt.value(design_system.COLORS["primary"][500]),
    )

    band = base.mark_errorband(extent="ci").encode(
        x=x_axis,
        y=y_axis,
        color=alt.value(design_system.COLORS["primary"][500]),
    )
    chart = band + line
    chart = (
        chart.properties(
            width=600,
            height=400,
            # title="Dormant neuron ratio across different number of tasks",
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

    figures_dir = pathlib.Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    chart.save(figures_dir / f"fig4_2_{args.width}.svg")


if __name__ == "__main__":
    main()
