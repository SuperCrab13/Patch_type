# Patch_Type
Use hovernet pretrained on PanNuke to classify patches of WSI into 6 types (no_label, neoplastic, inflammatory, connective, necros, non_neoplastic_epithelial). <br />
<br />
Hovernet assigns nucleis in each patch, takes the most frequently predicted nucleis type as the type of corresponding patch.

## Set Up Environment
``` 
pip install -r requirement.txt
```

## Get The Pretrained Model Weights
Click following links to download model weights
- [CoNSeP checkpoint](https://drive.google.com/file/d/1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P/view?usp=sharing)
- [PanNuke checkpoint](https://drive.google.com/file/d/1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR/view?usp=sharing)
- [MoNuSAC checkpoint](https://drive.google.com/file/d/13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV/view?usp=sharing)

And put model weights into ./data/weights

## Running the code
Suggest using the following data directory structure：<br />
```bash
DATA_DIRECTORY
    └──WSI_NAME/
          └──PATCHES
    └──weights/
          └──model weights
```

Running the following code to classify patches, edit the args if necessary. <br />
The data_dir should contain the patch of WSI, but not WSI file.

``` 
python get_patches.py --data_dir YOUR_DIRECTORY --cancer_type CANCER_TYPE
```
Result will be found in the ./patch_label directory. <br />
Visualization of the result

![](docs/visualize.jpg)

## Related Project:
- [HoverNet](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub)
- [HEAT](https://github.com/HKU-MedAI/WSI-HGNN)
