# Foundation and dependency source

The UGP implementations, training loops, losses, adapters, data interfaces and evaluation scripts are published in this repository. Their external foundation libraries remain available from their upstream repositories under the original licenses. Installed environments and third-party checkouts are not relabeled as project-authored code.

| Component | Public source | Use in the saved project |
|---|---|---|
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | Tensor operations, automatic differentiation, attention and training |
| Torchvision | [pytorch/vision](https://github.com/pytorch/vision) | Vision utilities in notebook environments |
| timm | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | The V7 notebook loads `timm/vit_large_patch16_dinov3.lvd1689m` and `timm/convnext_large.dinov3_lvd1689m` from the Hugging Face Hub |
| Utonia | [Pointcept/Utonia](https://github.com/Pointcept/Utonia) | Geometry feature extraction |
| WAFT | [princeton-vl/WAFT](https://github.com/princeton-vl/WAFT) | Optical-flow feature extraction |
| TartanAir tooling | [castacks/tartanairpy](https://github.com/castacks/tartanairpy) | Dataset access and recording utilities |
| ANYmal-D description | [ANYbotics/anymal_d_simple_description](https://github.com/ANYbotics/anymal_d_simple_description) | Robot geometry and kinematics |

These are the dependency identities referenced by the source, not a newly tested installation lockfile. Some historical installers clone the default branch without pinning a commit. Use the recorded environment and checkpoint/cache metadata when reproducing a particular run. Pretrained weights and raw data are obtained separately according to their access and redistribution terms.
