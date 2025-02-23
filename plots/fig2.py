"""Figure: Scaling chart for MT10/MT25/MT50

x-axis: number of parameters
y-axis: success rate
colour: benchmark
"""

import altair as alt
import design_system
import polars as ps
from get_data import get_metric

import pathlib


def main():
    wandb_entity = "reggies-phd-research"
    metric = "charts/mean_success_rate"

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

    data = ps.DataFrame(
        [
            {
                "Benchmark": benchmark,
                "Width": width,
                "Number of parameters": get_metric(
                    wandb_entity,
                    project(benchmark, width),
                    run_name(benchmark, width),
                    "actor_num_params",
                    source="config",
                )[0],
                "Success rate": get_metric(
                    wandb_entity,
                    project(benchmark, width),
                    run_name(benchmark, width),
                    metric,
                ),
            }
            for width in [256, 512, 1024, 2048, 4096]
            for benchmark in ["MT10", "MT25", "MT50"]
            if not (benchmark == "MT25" and width == 4096)  # FIXME:
        ]
    ).explode("Success rate")

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
        "mean(Success rate)",
        title="Success rate",
        scale=alt.Scale(domain=[0.5, 1]),
    )
    color_axis = alt.Color("Benchmark:N", title="Benchmark").scale(
        domain=["MT10", "MT25", "MT50"],
        range=[
            design_system.COLORS["primary"][500],
            design_system.COLORS["primary"][800],
            design_system.COLORS["grey"][800],
        ],
    )

    # Create the scaling chart
    base = alt.Chart(data).encode(
        x=x_axis,
        y=y_axis,
        color=color_axis,
    )

    line = base.mark_line(point=True).encode(
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
            width=600, height=400,
            # title="Scaling performance across MT10/MT25/MT50"
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
    chart.save(figures_dir / "fig2.svg")


if __name__ == "__main__":
    main()
