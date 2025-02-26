"""Figure: Actor vs Critic scaling

x-axis: timestep
y-axis: success rate
colour: both scaled, only actor scaled, only critic scaled, actor scaled down
"""

import pathlib

import altair as alt
import design_system
import polars as pl
from get_data import get_metric_history


def main():
    entity = "reggies-phd-research"
    metric = "charts/mean_success_rate"
    project = "mtrl-mt10-results"

    raw_data = [
        {
            "Variant": "Actor - 1024, Critic - 1024",
            "Success rate": get_metric_history(
                entity, project, "mt10_mtmhsac_v2_width_1024", metric
            ),
        },
        {
            "Variant": "Critic - 1024, Actor - 400",
            "Success rate": get_metric_history(
                entity, project, "mt10_mtmhsac_v2_wide_critic", metric
            ),
        },
        {
            "Variant": "Actor - 200, Critic - 1024",
            "Success rate": get_metric_history(
                entity, project, "mt10_mtmhsac_v2_wide_critic_shrunk_actor", metric
            ),
        },
        {
            "Variant": "Actor - 1024, Critic - 400",
            "Success rate": get_metric_history(
                entity, project, "mt10_mtmhsac_v2_wide_actor", metric
            ),
        },
        {
            "Variant": "Actor - 1024, Critic - 200",
            "Success rate": get_metric_history(
                entity, project, "mt10_mtmhsac_v2_wide_actor_shrunk_critic", metric
            ),
        },
    ]

    data = []
    for datum in raw_data:
        for i, run_data in enumerate(datum["Success rate"]):
            for step, value in run_data.items():
                data.append(
                    {
                        "Variant": datum["Variant"],
                        "Success rate": value,
                        "Timestep": step,
                        "Run": i,
                    }
                )
    data = pl.DataFrame(data)

    iq_data = data.group_by("Variant", "Timestep").agg(
        pl.col("Success rate").quantile(0.25).alias("q1"),
        pl.col("Success rate").quantile(0.75).alias("q3"),
    )
    data = data.join(
        iq_data,
        on=["Variant", "Timestep"],
        how="left",
    ).filter(
        (pl.col("Success rate") >= pl.col("q1"))
        & (pl.col("Success rate") <= pl.col("q3"))
    )

    max_timestep = 2e7
    x_axis = alt.X(
        "Timestep:Q",
        scale=alt.Scale(domain=[300_000, max_timestep]),
        title="Number of environment steps",
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
        scale=alt.Scale(domain=[0.2, 1]),
    )
    color_axis = alt.Color(
        "Variant:N",
        title="Network widths",
        legend=alt.Legend(orient="bottom-right", symbolOpacity=1.0, symbolSize=50),
    ).scale(
        domain=[item["Variant"] for item in raw_data],
        range=[
            design_system.COLORS["primary"][500],
            design_system.COLORS["primary"][700],
            design_system.COLORS["primary"][800],
            design_system.COLORS["grey"][500],
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
            # title=f"Success rate throughout training across scaling variations in SAC",
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
            labelLimit=200,
        )
    )

    figures_dir = pathlib.Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    chart.save(figures_dir / "fig5.svg")


if __name__ == "__main__":
    main()
