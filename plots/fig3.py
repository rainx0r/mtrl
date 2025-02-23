"""Figure: Success rate throughout training for one of MT10/MT25/MT50, for different scales

x-axis: timestep
y-axis: success rate
colour: number of parameters
"""

import pathlib
import polars as pl
import altair as alt
from get_data import get_metric, get_metric_history
import tyro
from dataclasses import dataclass
import design_system


@dataclass
class Args:
    benchmark: str = "MT10"


def main():
    args = tyro.cli(Args)

    entity = "reggies-phd-research"
    metric = "charts/mean_success_rate"

    if args.benchmark == "MT10":
        project = "mtrl-mt10-results"
    elif args.benchmark == "MT50":
        project = "mtrl-mt50-results"
    elif args.benchmark == "MT25":
        raise NotImplementedError
    else:
        raise ValueError

    def run_name(width: int):
        if args.benchmark == "MT10":
            return f"mt10_mtmhsac_v2_width_{width}"
        elif args.benchmark == "MT50":
            return f"mt50_mtmhsac_v2_{width}_width"
        elif args.benchmark == "MT25":
            raise NotImplementedError
        else:
            raise ValueError

    raw_data = [
        {
            "Benchmark": args.benchmark,
            "Width": width,
            "Number of parameters": get_metric(
                entity,
                project,
                run_name(width),
                "actor_num_params",
                source="config",
            )[0],
            "Success rate": get_metric_history(
                entity, project, run_name(width), metric
            ),
        }
        for width in [256, 512, 1024, 2048, 4096]
    ]

    # Expand history data into individual rows
    data = []
    for datum in raw_data:
        for i, run_data in enumerate(datum["Success rate"]):
            for step, value in run_data.items():
                data.append(
                    {
                        "Benchmark": datum["Benchmark"],
                        "Width": datum["Width"],
                        "Number of parameters": datum["Number of parameters"],
                        "Success rate": value,
                        "Timestep": step,
                        "Run": i,
                    }
                )
    data = pl.DataFrame(data)

    if args.benchmark == "MT10":
        max_timestep = 2e7
    else:
        max_timestep = 1e8
    x_axis = alt.X(
        "Timestep:Q",
        scale=alt.Scale(domain=[0, max_timestep]),
        title="Timestep",
        axis=alt.Axis(
            format="~s",
            labelExpr="datum.value >= 1000000 ? format(datum.value / 1000000, '.0f') + 'M' : datum.value >= 1000 ? format(datum.value / 1000, '.0f') + 'K' : datum.value",
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
        ),
    )
    y_axis = alt.Y(
        "mean(Success rate):Q",
        title="Success rate",
        scale=alt.Scale(domain=[0.1, 1]),
    )
    color_axis = alt.Color("Width:N", title="Width").scale(
        domain=[256, 512, 1024, 2048, 4096],
        range=[
            design_system.COLORS["primary"][400],
            design_system.COLORS["primary"][500],
            design_system.COLORS["primary"][600],
            design_system.COLORS["primary"][700],
            design_system.COLORS["primary"][800],
        ],
    )

    base = alt.Chart(data).encode(
        x=x_axis,
        y=y_axis,
        color=color_axis,
    )

    line = base.mark_line(clip=True, interpolate="basis-open").encode(
        x=x_axis,
        y=y_axis,
    )

    band = base.mark_errorband(clip=True, extent="ci", interpolate="basis-open").encode(
        x=x_axis,
        y=y_axis,
    )
    chart = band + line
    chart = (
        chart.properties(
            width=600,
            height=400,
            title=f"Success rate throughout training across different scales for {args.benchmark}",
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
    chart.save(figures_dir / f"fig3_{args.benchmark.lower()}.svg")


if __name__ == "__main__":
    main()
