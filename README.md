
# RNAProteinLoss function for RNA deeplearning methods(PyTorch)

This is a Huden`s RNAProteinLoss designed for deep learning models. Most of deep learning models will change the codon of each RNA when generating the RNA sequence. Therefore, I designed this loss function to regulate the generated RNA codon being within the right space (same protein). And CAI and tAI optimization have been applied in CAIloss and tAIloss.

Still under development...


## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN + torch >1.10 (Sorry but Compulsory, this is for deeplearning methods and we use torch in the research)

## Getting Started
### Installation
No comment...

### Usgae

```
from RPLoss import ...
import RPLoss as ...
```
