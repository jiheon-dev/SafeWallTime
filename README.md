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
