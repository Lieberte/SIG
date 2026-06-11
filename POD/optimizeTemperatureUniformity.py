import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import PodConfig


def buildMlp(inputDim: int, outputDim: int, hidden: int, layers: int):
    modules = []
    lastDim = inputDim
    for _ in range(max(1, int(layers))):
        modules.extend([torch.nn.Linear(lastDim, hidden), torch.nn.Tanh()])
        lastDim = hidden
    modules.append(torch.nn.Linear(lastDim, outputDim))
    return torch.nn.Sequential(*modules)


def loadPodNn(modelPath: Path, device: torch.device) -> dict:
    data = np.load(modelPath, allow_pickle=True)
    regressionType = str(data["regressionType"][0])
    if regressionType != "mlp":
        raise ValueError(f"Expected an MLP POD model, got regressionType={regressionType}")
    mean = torch.tensor(data["mean"], dtype=torch.float32, device=device)
    PHI = torch.tensor(data["PHI"], dtype=torch.float32, device=device)
    paramMin = torch.tensor(data["paramMin"], dtype=torch.float32, device=device)
    paramRange = torch.tensor(data["paramRange"], dtype=torch.float32, device=device)
    coeffMean = torch.tensor(data["coeffMean"], dtype=torch.float32, device=device)
    coeffStd = torch.tensor(data["coeffStd"], dtype=torch.float32, device=device)
    params = np.asarray(data["params"], dtype=np.float64)
    hidden = int(data["mlpHidden"][0])
    layers = int(data["mlpLayers"][0])
    model = buildMlp(3, PHI.shape[1], hidden, layers).to(device)
    state = {}
    for key in data.files:
        if key.startswith("mlp_"):
            state[key[4:]] = torch.tensor(data[key], dtype=torch.float32, device=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return {
        "mean": mean,
        "PHI": PHI,
        "paramMin": paramMin,
        "paramRange": paramRange,
        "coeffMean": coeffMean,
        "coeffStd": coeffStd,
        "params": params,
        "model": model,
    }


def makeTimeGrid(params: np.ndarray, timeStep: float | None) -> np.ndarray:
    times = np.unique(params[:, 0])
    times = np.sort(times)
    if timeStep is None or timeStep <= 0:
        return times
    return np.arange(float(times.min()), float(times.max()) + 0.5 * timeStep, timeStep)


def predictTemperature(pod: dict, times: torch.Tensor, TPreheat: torch.Tensor, TH2O2: torch.Tensor) -> torch.Tensor:
    P = torch.stack([
        times,
        TPreheat.expand_as(times),
        TH2O2.expand_as(times),
    ], dim=1)
    PN = (P - pod["paramMin"]) / pod["paramRange"]
    coeffNorm = pod["model"](PN)
    coeff = coeffNorm * pod["coeffStd"] + pod["coeffMean"]
    return pod["mean"] + pod["PHI"] @ coeff.T


def summarizeTemperature(T: torch.Tensor) -> dict:
    with torch.no_grad():
        spatialMean = T.mean(dim=0)
        spatialStd = T.std(dim=0)
        spatialMin = T.min(dim=0).values
        spatialMax = T.max(dim=0).values
        spatialRange = spatialMax - spatialMin
    return {
        "meanStdK": float(spatialStd.mean().cpu()),
        "maxStdK": float(spatialStd.max().cpu()),
        "meanRangeK": float(spatialRange.mean().cpu()),
        "maxRangeK": float(spatialRange.max().cpu()),
        "minTemperatureK": float(spatialMin.min().cpu()),
        "maxTemperatureK": float(spatialMax.max().cpu()),
        "meanTemperatureK": float(spatialMean.mean().cpu()),
    }


def makePerTimeRows(times: np.ndarray, T: torch.Tensor) -> list[dict]:
    with torch.no_grad():
        mean = T.mean(dim=0).detach().cpu().numpy()
        std = T.std(dim=0).detach().cpu().numpy()
        minVal = T.min(dim=0).values.detach().cpu().numpy()
        maxVal = T.max(dim=0).values.detach().cpu().numpy()
    rows = []
    for i, timeValue in enumerate(times):
        rows.append({
            "time": float(timeValue),
            "meanK": float(mean[i]),
            "stdK": float(std[i]),
            "minK": float(minVal[i]),
            "maxK": float(maxVal[i]),
            "rangeK": float(maxVal[i] - minVal[i]),
        })
    return rows


def plotComparison(times: np.ndarray, baselineRows: list[dict], optimizedRows: list[dict], outputPath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    baseStd = [row["stdK"] for row in baselineRows]
    optStd = [row["stdK"] for row in optimizedRows]
    baseRange = [row["rangeK"] for row in baselineRows]
    optRange = [row["rangeK"] for row in optimizedRows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, baseStd, "o-", label="baseline")
    axes[0].plot(times, optStd, "s--", label="optimized")
    axes[0].set_ylabel("std(T) K")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(times, baseRange, "o-", label="baseline")
    axes[1].plot(times, optRange, "s--", label="optimized")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("max(T)-min(T) K")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath, dpi=160)
    plt.close(fig)


def main() -> None:
    config = PodConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelPath", type=Path, default=config.outputPath)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "optimization" / "temperatureUniformity.json")
    parser.add_argument("--nIter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--timeStep", type=float, default=0.0)
    parser.add_argument("--rangeWeight", type=float, default=0.0)
    parser.add_argument("--targetMinK", type=float, default=0.0)
    parser.add_argument("--targetWeight", type=float, default=10.0)
    parser.add_argument("--baselinePreheatK", type=float, default=0.0)
    parser.add_argument("--baselineH2O2K", type=float, default=0.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pod = loadPodNn(args.modelPath, device)
    params = pod["params"]
    timesNp = makeTimeGrid(params, args.timeStep)
    times = torch.tensor(timesNp, dtype=torch.float32, device=device)
    pMin = pod["paramMin"].detach().cpu().numpy()
    pRange = pod["paramRange"].detach().cpu().numpy()
    preheatMin = float(pMin[1])
    preheatMax = float(pMin[1] + pRange[1])
    h2o2Min = float(pMin[2])
    h2o2Max = float(pMin[2] + pRange[2])
    basePreheat = args.baselinePreheatK if args.baselinePreheatK > 0 else 0.5 * (preheatMin + preheatMax)
    baseH2O2 = args.baselineH2O2K if args.baselineH2O2K > 0 else 0.5 * (h2o2Min + h2o2Max)
    raw = torch.zeros(2, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=args.lr)
    best = {"loss": float("inf"), "raw": None}
    for _ in range(args.nIter):
        optimizer.zero_grad()
        s = torch.sigmoid(raw)
        TPreheat = torch.tensor(preheatMin, dtype=torch.float32, device=device) + s[0] * (preheatMax - preheatMin)
        TH2O2 = torch.tensor(h2o2Min, dtype=torch.float32, device=device) + s[1] * (h2o2Max - h2o2Min)
        T = predictTemperature(pod, times, TPreheat, TH2O2)
        spatialStd = T.std(dim=0)
        spatialRange = T.max(dim=0).values - T.min(dim=0).values
        loss = spatialStd.mean() + args.rangeWeight * spatialRange.mean()
        if args.targetMinK > 0:
            minPenalty = torch.relu(args.targetMinK - T.min(dim=0).values)
            loss = loss + args.targetWeight * (minPenalty ** 2).mean()
        loss.backward()
        optimizer.step()
        lossValue = float(loss.detach().cpu())
        if lossValue < best["loss"]:
            best = {"loss": lossValue, "raw": raw.detach().clone()}
    rawBest = best["raw"]
    sBest = torch.sigmoid(rawBest)
    optPreheat = torch.tensor(preheatMin, dtype=torch.float32, device=device) + sBest[0] * (preheatMax - preheatMin)
    optH2O2 = torch.tensor(h2o2Min, dtype=torch.float32, device=device) + sBest[1] * (h2o2Max - h2o2Min)
    baselineT = predictTemperature(
        pod,
        times,
        torch.tensor(basePreheat, dtype=torch.float32, device=device),
        torch.tensor(baseH2O2, dtype=torch.float32, device=device),
    )
    optimizedT = predictTemperature(pod, times, optPreheat, optH2O2)
    baselineSummary = summarizeTemperature(baselineT)
    optimizedSummary = summarizeTemperature(optimizedT)
    baselineRows = makePerTimeRows(timesNp, baselineT)
    optimizedRows = makePerTimeRows(timesNp, optimizedT)
    figurePath = args.output.with_suffix(".png")
    plotComparison(timesNp, baselineRows, optimizedRows, figurePath)
    result = {
        "modelPath": str(args.modelPath),
        "boundsK": {
            "TPreheat": [preheatMin, preheatMax],
            "TH2O2": [h2o2Min, h2o2Max],
        },
        "baselineK": {
            "TPreheat": float(basePreheat),
            "TH2O2": float(baseH2O2),
        },
        "optimizedK": {
            "TPreheat": float(optPreheat.detach().cpu()),
            "TH2O2": float(optH2O2.detach().cpu()),
        },
        "objective": {
            "nIter": args.nIter,
            "lr": args.lr,
            "rangeWeight": args.rangeWeight,
            "targetMinK": args.targetMinK,
            "targetWeight": args.targetWeight,
            "bestLoss": best["loss"],
        },
        "baselineSummary": baselineSummary,
        "optimizedSummary": optimizedSummary,
        "improvement": {
            "meanStdReductionPercent": 100.0 * (baselineSummary["meanStdK"] - optimizedSummary["meanStdK"]) / max(baselineSummary["meanStdK"], 1e-12),
            "meanRangeReductionPercent": 100.0 * (baselineSummary["meanRangeK"] - optimizedSummary["meanRangeK"]) / max(baselineSummary["meanRangeK"], 1e-12),
        },
        "perTime": {
            "baseline": baselineRows,
            "optimized": optimizedRows,
        },
        "figure": str(figurePath),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "figure": str(figurePath),
        "boundsK": result["boundsK"],
        "baselineK": result["baselineK"],
        "optimizedK": result["optimizedK"],
        "baselineSummary": baselineSummary,
        "optimizedSummary": optimizedSummary,
        "improvement": result["improvement"],
    }, indent=2))


if __name__ == "__main__":
    main()
