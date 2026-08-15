# SafeWallTime

SafeWallTime is a Python framework for uncertainty-aware runtime prediction in
HPC job scheduling. 

## Install

The framework itself has only NumPy and pandas dependencies. Install LightGBM
to use the built-in multi-quantile predictor:

```bash
python -m pip install -e ".[lightgbm]"
python -m pytest
```

## UARP baseline

```python
from safewalltime import (
    AdaptiveMargin,
    MultiQuantilePredictor,
    SWFLoader,
    UARPPipeline,
    UARPPreprocessor,
)

pipeline = UARPPipeline(
    loader=SWFLoader(min_runtime=10),
    preprocessor=UARPPreprocessor(),
    predictor=MultiQuantilePredictor(quantiles=[0.50, 0.99]),
    margin=AdaptiveMargin(alpha=0.2, beta=0.5),
)
result = pipeline.run("path/to/trace.swf")
```


## Citation

If you use this code, please cite the following papers:

```
@article{choi2026uarp,
  title={UARP: uncertainty-aware runtime prediction for preventing scheduler termination under Wallclock constraints in HPC},
  author={Choi, Jiheon and Oh, Sangyoon},
  journal={The Journal of Supercomputing},
  volume={82},
  number={5},
  pages={292},
  year={2026},
  publisher={Springer}
}
```

```
@InProceedings{choi2026scqr,
author="Choi, Jiheon and Oh, Sangyoon",
title="S-CQR: Stratified Calibration for Runtime Prediction in HPC Backfill Scheduling",
booktitle="Euro-Par 2026: Parallel Processing",
year="2027",
publisher="Springer Nature Switzerland",
pages="316--330",
isbn="978-3-032-35251-4"
}
```
