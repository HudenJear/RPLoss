
# RNAProteinLoss function for RNA deeplearning methods(PyTorch)

This is a Huden`s RNAProteinLoss designed for deep learning models. Most of deep learning models will change the codon of each RNA when generating the RNA sequence. Therefore, I designed this loss function to regulate the generated RNA codon being within the right space (same protein). And CAI and tAI optimization have been applied in CAIloss and tAIloss.

Still under development...


## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN + torch >1.10 (Sorry but Compulsory, this is for deeplearning methods and we use torch in the research)

## Getting Started
### Installation

No comment...For now, you can use it directly in your model loss by import it from this dictionary

### Usgae

```
from RPLoss import ...
import RPLoss as ...
```

The CAILoss and tAILoss need a specified speceis when initializing, the other two need not.
The loss weight of RPLoss shall be very high, and we recommand the ratio of 10000:1:1:0.1 when using the RP/CAI/tAI/MFE Loss due to their function and value scale.
