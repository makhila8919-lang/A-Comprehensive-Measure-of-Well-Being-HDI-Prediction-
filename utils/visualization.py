import matplotlib.pyplot as plt
import seaborn as sns


def plot_hdi_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(x="HDI Category", data=df)

    plt.title("HDI Category Distribution")

    plt.xlabel("HDI Category")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("static/charts/hdi_distribution.png")

    plt.close()


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    plt.figure(figsize=(8, 5))

    sns.barplot(x=importance, y=feature_names)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("static/charts/feature_importance.png")

    plt.close()


def plot_correlation(df):

    plt.figure(figsize=(8, 6))

    sns.heatmap(

        df.corr(numeric_only=True),

        annot=True,

        cmap="Blues"

    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig("static/charts/correlation.png")

    plt.close()
