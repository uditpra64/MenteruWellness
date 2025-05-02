from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

weekday_en_to_jp = {
    0: "月曜日",
    1: "火曜日",
    2: "水曜日",
    3: "木曜日",
    4: "金曜日",
    5: "土曜日",
    6: "日曜日",
}
weekday_str_en_to_jp = {
    "Monday": "月曜日",
    "Tuesday": "火曜日",
    "Wednesday": "水曜日",
    "Thursday": "木曜日",
    "Friday": "金曜日",
    "Saturday": "土曜日",
    "Sunday": "日曜日",
}


def plot_scatter(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str = "",
    ylabel: str = "",
    xticks: Union[list, None] = None,
    alpha: float = 1,
    title: Union[str, None] = None,
    color: str = None,
):
    """散布図を作成。"""
    #     sns.scatterplot(data=data, x=x, y=y, alpha=alpha, ax=ax, color=color)
    sns.regplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        scatter_kws=dict(color=color, facecolor="none", alpha=0.5, s=8),
        line_kws=dict(color="blue", linewidth=2),
    )
    ax.plot(
        np.arange(np.max(data[x])),
        np.arange(np.max((data[x]))),
        linewidth=2,
        color="grey",
        alpha=1,
    )
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlim([xticks[0], xticks[-1]])
    ax.set_ylim([xticks[0], xticks[-1]])
    ax.set_aspect(1, adjustable="box")
    ax.set_title(title)


def plot_scatters_train_valid(
    score_evaluator_train, score_evaluator_val, xticks, CASE_NAME, feature_name
):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_scatter(
        ax1,
        score_evaluator_train.df,
        x="true",
        y="pred",
        xlabel="実績値",
        ylabel="予測値",
        xticks=xticks,
        alpha=0.5,
        title=f"{feature_name} train ({CASE_NAME})",
        color="tab:blue",
    )
    plot_scatter(
        ax2,
        score_evaluator_val.df,
        x="true",
        y="pred",
        xlabel="実績値",
        ylabel="予測値",
        xticks=xticks,
        alpha=0.5,
        title=f"{feature_name} val ({CASE_NAME})",
        color="tab:blue",
    )
    fig.tight_layout()


def plot_ts(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    xticks: Union[list, None] = None,
    color=None,
    alpha: float = 1,
    title: Union[str, None] = None,
    label: Union[str, None] = None,
    yticks: Union[list, None] = None,
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
):
    """時系列グラフを作成。"""
    plot = sns.lineplot(
        data=data, x=x, y=y, alpha=alpha, ax=ax, label=label, color=color
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([data[x].iloc[0], data[x].iloc[-1]])
    if yticks is not None:
        ax.set_yticks(yticks)
    return plot


def plot_ts_area(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    xticks: Union[list, None] = None,
    color=None,
    alpha: float = 1,
    title: Union[str, None] = None,
    label: Union[str, None] = None,
    yticks: Union[list, None] = None,
    xlabel: Union[str, None] = None,
    ylabel: Union[str, None] = None,
):
    """時系列グラフを作成。"""
    plot = sns.lineplot(
        data=data,
        x=x,
        y=y,
        alpha=alpha,
        ax=ax,
        label=label,
        color=color,
        linestyle="--",
    )
    x_ = data["datetime"].values
    y1 = [0] * len(x_)
    y2 = data["temp"].values
    ax.fill_between(x_, y1, y2, color="green", alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([data[x].iloc[0], data[x].iloc[-1]])
    if yticks is not None:
        ax.set_yticks(yticks)
    return plot


def plot_ts_train_valid(valid, output_features, yticks, feature_name, CASE_NAME):
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="true",
        label="true",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="grey",
    )
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="pred",
        label="pred",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="tab:orange",
    )

    # import matplotlib.dates as mdates
    # valid['datetime'] = pd.to_datetime(valid['datetime'])

    # # Set the locator for major ticks to one tick per month
    # ax.xaxis.set_major_locator(mdates.MonthLocator())

    # # Set the formatter for major ticks to show the date as 'YYYY-MM'
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # # Rotate the labels for better legibility
    # ax.get_xticklabels()

    valid.loc[:, "weekday_str"] = valid.loc[:, "datetime"].apply(
        lambda x: x.strftime("%A")
    )
    valid.loc[:, "weekday_str"] = valid.loc[:, "weekday_str"].replace(
        weekday_str_en_to_jp
    )

    xticks = valid.loc[:, "date"] + "\n" + "(" + valid.loc[:, "weekday_str"] + ")"
    xticks = xticks.unique().tolist()
    xticks.sort()
    _ = ax.set_xticklabels(xticks)


def plot_ts_valid_with_temp(valid, output_features, yticks, title=None):
    fig, ax = plt.subplots(figsize=(16, 4))
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="true",
        label="true",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="grey",
    )
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="pred",
        label="pred",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="tab:orange",
    )

    yticks_temp = [i for i in range(0, 51, 10)]
    ax2 = ax.twinx()
    plot_ts_area(
        ax2,
        valid,
        x="datetime",
        y="temp",
        label="temp",
        color="tab:green",
        yticks=yticks_temp,
        xlabel="datetime",
        ylabel="temp",
    )
    ax2.grid(False)
    ax2.set_ylabel("temp", rotation=270, labelpad=20)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")


def plot_scores_by_month(df, feature_name, CASE_NAME):
    xticks = range(1, 13)

    def set_format(ax1, ax2):
        ax1.set_xticks(xticks)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)
        ax2.grid(False)
        ax1.set_title(f"{feature_name} 評価期間: 2022-04-01 ～2023-03-31 ({CASE_NAME})")
        legends = plot1 + plot2
        labels = [legend.get_label() for legend in legends]
        ax1.legend(legends, labels, loc=0)

    # Determine max values for MAE and RMSE to decide on y-axis limits
    max_mae = df["mean_absolute_error"].max()
    max_rmse = (
        df["root_mean_squared_error"].max() if "root_mean_squared_error" in df else None
    )

    # Adjust y-axis limits based on the max values
    mae_limit = max(5, max_mae * 1.1) if max_mae > 10 else 3
    rmse_limit = max(5, max_rmse * 1.1) if max_rmse and max_rmse > 10 else 4

    # Plot for MAE and MAPE
    fig, ax = plt.subplots(figsize=(16, 4))
    plot1 = ax.plot(
        df["month"],
        df["mean_absolute_error"],
        marker="o",
        color="tab:blue",
        linestyle="--",
        label="MAE",
    )
    ax2 = ax.twinx()
    plot2 = ax2.plot(
        df["month"],
        df["mean_absolute_percentage_error"],
        marker="s",
        color="tab:orange",
        linestyle="--",
        label="MAPE",
    )
    set_format(ax, ax2)
    ax.set_ylim([0, mae_limit])
    ax2.set_ylim([0, 60])
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("MAPE", labelpad=20, rotation=270)

    # Check if RMSE data is available before plotting
    if max_rmse:
        fig, ax = plt.subplots(figsize=(16, 4))
        plot1 = ax.plot(
            df["month"],
            df["mean_absolute_error"],
            marker="o",
            color="tab:blue",
            linestyle="--",
            label="MAE",
        )
        ax2 = ax.twinx()
        plot2 = ax2.plot(
            df["month"],
            df["root_mean_squared_error"],
            marker="s",
            color="tab:orange",
            linestyle="--",
            label="RMSE",
        )
        set_format(ax, ax2)
        ax.set_ylim([0, mae_limit])
        ax2.set_ylim([0, rmse_limit])
        ax.set_xlabel("Month")
        ax.set_ylabel("MAE")
        ax2.set_ylabel("RMSE", labelpad=20, rotation=270)


def plot_mae_mape_by_month(df, feature_name, CASE_NAME, y_limits):
    fig, ax = plt.subplots(figsize=(16, 4))
    xticks = range(1, 13)

    # Plot MAE
    plot1 = ax.plot(
        df["month"],
        df["mean_absolute_error"],
        marker="o",
        color="tab:blue",
        linestyle="--",
        label="MAE",
    )

    # Create a second y-axis for MAPE
    ax2 = ax.twinx()
    plot2 = ax2.plot(
        df["month"],
        df["mean_absolute_percentage_error"],
        marker="s",
        color="tab:orange",
        linestyle="--",
        label="MAPE",
    )

    # Format and settings
    ax.set_xticks(xticks)
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax2.grid(False)
    ax.set_title(f"{feature_name} 評価期間: 2022-04-01 ～2023-03-31 ({CASE_NAME})")

    # Combine legends from both plots
    legends = plot1 + plot2
    labels = [legend.get_label() for legend in legends]
    ax.legend(legends, labels, loc=0)

    # Set y-axis limits
    ax.set_ylim(y_limits[0])
    ax2.set_ylim(y_limits[1])

    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("MAPE", labelpad=20, rotation=270)

    plt.show()


def plot_mae_rmse_by_month(df, feature_name, CASE_NAME, y_limits):
    fig, ax = plt.subplots(figsize=(16, 4))
    xticks = range(1, 13)

    # Plot MAE
    plot1 = ax.plot(
        df["month"],
        df["mean_absolute_error"],
        marker="o",
        color="tab:blue",
        linestyle="--",
        label="MAE",
    )

    # Create a second y-axis for RMSE
    ax2 = ax.twinx()
    plot2 = ax2.plot(
        df["month"],
        df["root_mean_squared_error"],
        marker="s",
        color="tab:orange",
        linestyle="--",
        label="RMSE",
    )

    # Format and settings
    ax.set_xticks(xticks)
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax2.grid(False)
    ax.set_title(f"{feature_name} 評価期間: 2022-04-01 ～2023-03-31 ({CASE_NAME})")

    # Combine legends from both plots
    legends = plot1 + plot2
    labels = [legend.get_label() for legend in legends]
    ax.legend(legends, labels, loc=0)

    # Set y-axis limits
    ax.set_ylim(y_limits[0])
    ax2.set_ylim(y_limits[1])

    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("RMSE", labelpad=20, rotation=270)

    plt.show()


def plot_box_month(df, xlabel=None, ylabel=None, title=""):
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.boxplot(
        data=df,
        x="month",
        y="absolute_percentage_error(%)",
        whis=[0, 100],
        showmeans=True,
        meanprops={"marker": "^", "markeredgecolor": "black", "markersize": 10},
        ax=ax,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.boxplot(
        data=df,
        x="month",
        y="absolute_percentage_error(%)",
        whis=[0, 100],
        showmeans=True,
        meanprops={"marker": "^", "markeredgecolor": "black", "markersize": 10},
    )
    sns.stripplot(
        data=df,
        x="month",
        y="absolute_percentage_error(%)",
        edgecolor="black",
        linewidth=0.2,
        facecolor="none",
    )


def plot_box_weekday(df, xlabel=None, ylabel=None, title=""):
    fig, ax = plt.subplots(figsize=(16, 4))
    df["weekday"] = df["weekday"].replace(weekday_en_to_jp)
    sns.boxplot(
        data=df,
        x="weekday",
        y="absolute_percentage_error(%)",
        whis=[0, 100],
        showmeans=True,
        order=list(weekday_en_to_jp.values()),
        meanprops={"marker": "^", "markeredgecolor": "black", "markersize": 10},
    )
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.boxplot(
        data=df,
        x="weekday",
        y="absolute_percentage_error(%)",
        whis=[0, 100],
        showmeans=True,
        order=list(weekday_en_to_jp.values()),
        meanprops={"marker": "^", "markeredgecolor": "black", "markersize": 10},
    )
    sns.stripplot(
        data=df,
        x="weekday",
        y="absolute_percentage_error(%)",
        edgecolor="black",
        linewidth=0.2,
        facecolor="none",
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_scatter_element(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue=None,
    xlabel: str = "",
    ylabel: str = "",
    xticks: Union[list, None] = None,
    alpha: float = 1,
    title: Union[str, None] = None,
    color: str = None,
    cms=None,
):
    """散布図を作成。"""
    #     sns.scatterplot(data=data, x=x, y=y, alpha=alpha, ax=ax, color=color)
    sns.regplot(
        data=data,
        x=x,
        y=y,
        ax=ax,
        scatter_kws=dict(color=color, facecolor="none", alpha=0.5, s=8),
        scatter=False,
        line_kws=dict(color="blue", linewidth=2),
    )
    sns.scatterplot(data=data, hue=hue, palette=cms, x=x, y=y, alpha=0.6, s=16)
    ax.plot(
        np.arange(np.max(data[x])),
        np.arange(np.max((data[x]))),
        linewidth=2,
        color="grey",
        alpha=1,
    )
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # ax.set_xlim([min_,max_]);ax.set_ylim([min_,max_])
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlim([xticks[0], xticks[-1]])
    ax.set_ylim([xticks[0], xticks[-1]])
    ax.set_aspect(1, adjustable="box")
    ax.set_title(title)


def plot_ts_valid_formatted(
    valid,
    output_features,
    yticks,
    title=None,
    feature_name="",
    unit=None,
    temperature=False,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="true",
        label="実績値",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="grey",
    )
    plot_ts(
        ax,
        valid,
        x="datetime",
        y="pred",
        label="予測値",
        yticks=yticks,
        xlabel="datetime",
        ylabel=output_features[0],
        color="tab:orange",
    )
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(f"{feature_name}{unit}")

    valid.loc[:, "weekday_str"] = valid.loc[:, "datetime"].apply(
        lambda x: x.strftime("%A")
    )
    valid.loc[:, "weekday_str"] = valid.loc[:, "weekday_str"].replace(
        weekday_str_en_to_jp
    )
    valid["datetime"] = pd.to_datetime(valid["datetime"])

    valid.loc[:, "weekday_str"] = valid["datetime"].apply(lambda x: x.strftime("%A"))
    valid.loc[:, "weekday_str"] = valid.loc[:, "weekday_str"].replace(
        weekday_str_en_to_jp
    )
    xtick_labels = (
        valid["date"].astype(str) + "\n" + "(" + valid.loc[:, "weekday_str"] + ")"
    )
    xtick_labels = xtick_labels.unique().tolist()
    xtick_labels.sort()
    unique_dates = valid["datetime"].dt.date.unique()
    xtick_positions = [
        valid[valid["datetime"].dt.date == date].index[0] for date in unique_dates
    ]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    ax.legend(loc="upper right")

    if temperature:
        yticks2 = [i for i in range(0, 51, 10)]

        ax2 = ax.twinx()
        plot_ts_area(
            ax2,
            valid,
            x="datetime",
            y="temp",
            label="気温",
            color="tab:green",
            yticks=yticks,
            xlabel="datetime",
            ylabel="temp",
        )
        ax2.grid(False)
        ax2.set_ylabel("気温(℃)", rotation=270, labelpad=20)
        ax2.set_ylim([0, 50])
        _ = ax2.set_yticks(yticks2)

        valid.loc[:, "weekday_str"] = valid.loc[:, "datetime"].apply(
            lambda x: x.strftime("%A")
        )
        valid.loc[:, "weekday_str"] = valid.loc[:, "weekday_str"].replace(
            weekday_str_en_to_jp
        )
        xticks = valid.loc[:, "date"] + "\n" + "(" + valid.loc[:, "weekday_str"] + ")"
        xticks = xticks.unique().tolist()
        xticks.sort()
        _ = ax.set_xticklabels(xticks)
        ax.legend(loc="upper left")
        ax.set_title(title, fontsize=18)
        ax2.legend(loc="upper right")


def plot_scatter_formatted(score_evaluator_val, ticks, feature_name, unit, CASE_NAME):
    fig = plt.figure(figsize=(6, 8))
    ax2 = fig.add_subplot(1, 1, 1)
    xticks = ticks
    plot_scatter(
        ax2,
        score_evaluator_val.df,
        x="true",
        y="pred",
        xlabel="実績値",
        ylabel="予測値",
        xticks=xticks,
        alpha=0.5,
        title="val",
        color="tab:blue",
    )
    ax2.set_xlabel(f"実績値{unit}")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, fontsize=16)
    ax2.set_ylabel(f"予測値{unit}")
    ax2.set_yticks(xticks)
    ax2.set_yticklabels(xticks, fontsize=16)
    ax2.set_title(f"{feature_name} ({CASE_NAME})", fontsize=18)
    fig.tight_layout()


def plot_scatter_element_with_hue(
    score_evaluator_val, ticks, hue, feature_name, unit, CASE_NAME
):
    as_cmap = True
    if score_evaluator_val.df[hue].unique().shape[0] <= 2:
        as_cmap = False
    cms = sns.color_palette("rainbow", as_cmap=as_cmap)

    fig = plt.figure(figsize=(6, 8))
    ax2 = fig.add_subplot(1, 1, 1)
    xticks = ticks
    plot_scatter_element(
        ax2,
        score_evaluator_val.df,
        x="true",
        y="pred",
        hue=hue,
        xlabel="実績値",
        ylabel="予測値",
        xticks=xticks,
        alpha=0.5,
        title="val",
        color="tab:blue",
        cms=cms,
    )
    ax2.set_xlabel(f"実績値{unit}")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticks, fontsize=16)
    ax2.set_ylabel(f"予測値{unit}")
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(ticks, fontsize=16)
    ax2.set_title(f"{feature_name} ({CASE_NAME})", fontsize=18)
    fig.tight_layout()


def plot_monthly_scores(df, feature_name, CASE_NAME, figsize=(8, 4), y_limit=None):
    xticks = range(1, 13)
    xticklabels = [f"{i}月" for i in xticks]

    fig, ax = plt.subplots(figsize=figsize)
    (mae_plot,) = ax.plot(
        df["month"],
        df["mean_absolute_error"],
        marker="o",
        color="tab:blue",
        linestyle="--",
        label="MAE(GJ)",
    )
    ax2 = ax.twinx()
    (mape_plot,) = ax2.plot(
        df["month"],
        df["mean_absolute_percentage_error"],
        marker="s",
        color="tab:orange",
        linestyle="--",
        label="MAPE(%)",
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax2.grid(False)
    ax.set_ylim(
        [0, y_limit[0]]
    )  # Adjusted the y-axis limit for MAE 200, 101 and 3.0 and 101
    ax2.set_ylim(
        [0, y_limit[1]]
    )  # Adjusted the y-axis limit for MAPE 200, 101 and 3.0 and 101
    ax.set_xlabel("")
    ax.set_ylabel("MAE(GJ)")
    ax2.set_ylabel("MAPE(%)", labelpad=20, rotation=270)
    # Combine legends
    plots = [mae_plot, mape_plot]
    labels = [plot.get_label() for plot in plots]
    ax.legend(plots, labels, loc="upper right")
    ax.set_title(f"{feature_name} ({CASE_NAME})", fontsize=18)
    return fig, ax, ax2


def plot_feature_importance(model):
    """Plot feature importances for an XGBoost model."""
    importances = model.get_feature_importance()
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    features, importance_values = zip(*sorted_importances)

    plt.figure(figsize=(15, 10))
    bar_width = 0.5  # Width of the bars
    bar_spacing = 1.5  # Spacing between bars

    # Creating bars with specified width and spacing
    for i in range(len(importance_values)):
        plt.bar(i * bar_spacing, importance_values[i], width=bar_width)

    # Adjusting the x-ticks to match the bars
    plt.xticks(
        [i * bar_spacing for i in range(len(features))],
        features,
        rotation=45,
        ha="right",
    )

    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.title("Feature Importances in XGBoost Model")
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    plt.show()
