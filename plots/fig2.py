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
    data = ps.DataFrame(
        [
            {
                "Benchmark": "MT10",
                "Width": "256",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_256",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_256",
                    metric,
                ),
            },
            {
                "Benchmark": "MT10",
                "Width": "512",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_512",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_512",
                    metric,
                ),
            },
            {
                "Benchmark": "MT10",
                "Width": "1024",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_1024",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_1024",
                    metric,
                ),
            },
            {
                "Benchmark": "MT10",
                "Width": "2048",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_2048",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_2048",
                    metric,
                ),
            },
            {
                "Benchmark": "MT10",
                "Width": "4096",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_4096",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt10-results",
                    "mt10_mtmhsac_v2_width_4096",
                    metric,
                ),
            },
            {
                "Benchmark": "MT25",
                "Width": "256",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_256_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_256_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT25",
                "Width": "512",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_512_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_512_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT25",
                "Width": "1024",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_1024_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_1024_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT25",
                "Width": "2048",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_2048_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt25-results",
                    "mt25_mtmhsac_v2_2048_width",
                    metric,
                ),
            },
            # {
            #     "Benchmark": "MT25",
            #     "Width": "4096",
            # "Number of Parameters": get_metric(
            #     wandb_entity, "mtrl-mt50-results", "mt25_mtmhsac_v2_4096_width", "actor_num_params", source="config"
            # ),
            #     "Success Rate": get_metric(  # FIXME: This should be in the mt25 project, also not dnoe yet
            #         wandb_entity, "mtrl-mt50-results", "mt25_mtmhsac_v2_4096", metric
            #     ),
            # },
            {
                "Benchmark": "MT50",
                "Width": "256",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_256_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_256_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT50",
                "Width": "512",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_512_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_512_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT50",
                "Width": "1024",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_1024_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_1024_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT50",
                "Width": "2048",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_2048_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_2048_width",
                    metric,
                ),
            },
            {
                "Benchmark": "MT50",
                "Width": "4096",
                "Number of Parameters": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_4096_width",
                    "actor_num_params",
                    source="config",
                )[0],
                "Success Rate": get_metric(
                    wandb_entity,
                    "mtrl-mt50-results",
                    "mt50_mtmhsac_v2_4096_width",
                    metric,
                ),
            },
        ]
    ).explode("Success Rate")

    x_axis = alt.X(
        "Number of Parameters:Q",
        scale=alt.Scale(
            type="log",
            domain=[  # pyright: ignore [reportArgumentType]
                data["Number of Parameters"].min(),
                data["Number of Parameters"].max(),
            ],
        ),
        title="Number of Parameters",
        axis=alt.Axis(
            format="~s",
            labelExpr="datum.value >= 1000000 ? format(datum.value / 1000000, '.0f') + 'M' : datum.value >= 1000 ? format(datum.value / 1000, '.0f') + 'K' : datum.value",
            titleFont=design_system.PRIMARY_FONT,
            labelFont=design_system.SECONDARY_FONT,
        ),
    )
    y_axis = alt.Y(
        "mean(Success Rate)",
        title="Success Rate",
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
            width=600, height=400, title="Scaling Performance Across MT10/MT25/MT50"
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
