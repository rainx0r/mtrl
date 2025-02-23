"""Figure: dormant neuron ratio at the end of training for different scales, for MT10/MT25/MT50

x-axis: number of parameters
y-axis: dormant neuron ratio at the end of training
color: benchmark
"""

import pathlib

import altair as alt
import design_system
import polars as pl
from get_data import get_metric, get_metric_history


def main():
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
            "Width": width,
            "Number of parameters": get_metric(
                entity,
                project(benchmark, width),
                run_name(benchmark, width),
                "actor_num_params",
                source="config",
            )[0],
            "Dormant neuron ratio": get_metric_history(
                entity,
                project(benchmark, width),
                run_name(benchmark, width),
                metric,
            ),
        }
        for (benchmark, num_tasks) in [("MT10", 10), ("MT50", 50), ("MT25", 25)]
        for width in [256, 512, 1024, 2048]
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
    data = data.filter(
        pl.col("Timestep") == pl.col("Timestep").max().over("Number of tasks", "Number of parameters")
    ).drop("Timestep")

    print(data)

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
        ),
    )
    y_axis = alt.Y(
        "mean(Dormant neuron ratio):Q",
        title="Dormant neuron ratio (%)",
        scale=alt.Scale(domain=[0, 50]),
    )
    color_axis = alt.Color("Number of tasks:O", title="Number of tasks").scale(
        domain=[10, 25, 50],
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

    line = base.mark_line(clip=True, point=True).encode(
        x=x_axis,
        y=y_axis,
    )

    band = base.mark_errorband(extent="ci").encode(
        x=x_axis,
        y=y_axis,
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
    chart.save(figures_dir / "fig4_3.svg")


if __name__ == "__main__":
    main()
