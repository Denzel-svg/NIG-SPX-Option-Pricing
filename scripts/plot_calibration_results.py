"""Create report-ready figures from calibration result CSV files."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class CalibrationFigureGenerator:
    """Generate price-fit, residual and error-comparison figures."""

    def __init__(self, fit_path: Path, expiry_path: Path, output_dir: Path) -> None:
        self.fit_path = fit_path
        self.expiry_path = expiry_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load quote-level and maturity-level outputs and validate their schemas."""
        fit = pd.read_csv(self.fit_path)
        expiry = pd.read_csv(self.expiry_path)
        required_fit = {"Expiry", "Strike", "Spot", "Market_Price", "Model_Price", "BS_Model_Price", "Pricing_Error"}
        required_expiry = {"Expiry", "NIG_RMSE", "BS_RMSE", "NIG_ARPE", "BS_ARPE"}
        if missing := required_fit.difference(fit.columns):
            raise ValueError(f"Quote-level results are missing columns: {sorted(missing)}")
        if missing := required_expiry.difference(expiry.columns):
            raise ValueError(f"Maturity-level results are missing columns: {sorted(missing)}")
        if fit.empty or expiry.empty:
            raise ValueError("Calibration result files must not be empty.")
        fit["Moneyness"] = fit["Strike"] / fit["Spot"]
        fit["Expiry"] = fit["Expiry"].astype(str)
        expiry["Expiry"] = expiry["Expiry"].astype(str)
        return fit, expiry

    def plot_prices_by_expiry(self, fit: pd.DataFrame) -> None:
        """Compare market, NIG and common-volatility Black-Scholes prices by expiry."""
        expiries = sorted(fit["Expiry"].unique())
        fig, axes = plt.subplots(len(expiries), 1, figsize=(10, 3.1 * len(expiries)))
        if len(expiries) == 1:
            axes = [axes]
        for axis, expiry in zip(axes, expiries):
            sample = fit.loc[fit["Expiry"] == expiry].sort_values("Strike")
            axis.scatter(sample["Strike"], sample["Market_Price"], s=18, facecolors="none",
                         edgecolors="black", linewidths=0.8, label="Market", zorder=3)
            axis.plot(sample["Strike"], sample["Model_Price"], color="tab:blue", linewidth=1.6, label="NIG")
            axis.plot(sample["Strike"], sample["BS_Model_Price"], color="tab:orange", linewidth=1.6,
                      linestyle="--", label="Black-Scholes")
            axis.set_title(expiry)
            axis.set_ylabel("Call price (index points)")
            axis.grid(True, alpha=0.25)
            axis.legend()
        axes[-1].set_xlabel("Strike")
        fig.suptitle("Market and model call prices by maturity")
        fig.tight_layout()
        fig.savefig(self.output_dir / "market_nig_bs_by_expiry.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_residuals_by_moneyness(self, fit: pd.DataFrame) -> None:
        """Plot NIG model-minus-market residuals against strike moneyness."""
        fig, axis = plt.subplots(figsize=(9, 5))
        for expiry, sample in fit.groupby("Expiry"):
            axis.scatter(sample["Moneyness"] * 100, sample["Pricing_Error"], s=20, alpha=0.75, label=expiry)
        axis.axhline(0.0, linewidth=1, linestyle="--")
        axis.set_xlabel("Moneyness (strike / spot, %)")
        axis.set_ylabel("NIG pricing error (model - market)")
        axis.set_title("NIG pricing residuals across moneyness")
        axis.grid(True, alpha=0.25)
        axis.legend(title="Expiry", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "nig_residuals_by_moneyness.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def plot_error_comparison(self, expiry: pd.DataFrame) -> None:
        """Compare absolute and relative NIG/Black-Scholes errors by expiry."""
        labels = expiry["Expiry"].tolist()
        positions = list(range(len(labels)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        width = 0.36
        axes[0].bar([p - width / 2 for p in positions], expiry["NIG_RMSE"], width, label="NIG")
        axes[0].bar([p + width / 2 for p in positions], expiry["BS_RMSE"], width, label="Black-Scholes")
        axes[0].set_title("RMSE by maturity")
        axes[0].set_ylabel("RMSE (index points)")
        axes[1].bar([p - width / 2 for p in positions], expiry["NIG_ARPE"], width, label="NIG")
        axes[1].bar([p + width / 2 for p in positions], expiry["BS_ARPE"], width, label="Black-Scholes")
        axes[1].set_title("ARPE by maturity")
        axes[1].set_ylabel("ARPE (%)")
        for axis in axes:
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "nig_vs_bs_errors_by_expiry.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def run(self) -> None:
        """Generate the complete set of Section 7 figures."""
        fit, expiry = self.load_data()
        self.plot_prices_by_expiry(fit)
        self.plot_residuals_by_moneyness(fit)
        self.plot_error_comparison(expiry)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    CalibrationFigureGenerator(
        fit_path=project_root / "results" / "calibration" / "nig_vs_black_scholes_fit.csv",
        expiry_path=project_root / "results" / "calibration" / "nig_vs_black_scholes_by_expiry.csv",
        output_dir=project_root / "results" / "figures",
    ).run()
