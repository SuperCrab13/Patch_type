# Patch_Type
Use hovernet pretrained on PanNuke to classify patches of WSI into 6 types (no_label, neoplastic, inflammatory, connective, necros, non_neoplastic_epithelial). <br />
<br />
Hovernet assigns nucleis in each patch, takes the most frequently predicted nucleis type as the type of corresponding patch.

## Set Up Environment
``` 
pip install -r requirement.txt
```

## Running the code
Running the following code to classify patches, edit the args if necessary. <br />
The data_dir should contain the patch of WSI, but not WSI file.

``` 
python get_patches.py --data_dir YOUR_DIRECTORY --cancer_type CANCER_TYPE
```

Visualization of the result

![](docs/visualize.jpeg)

## Related Project:
- [HoverNet](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub)
- [HEAT](https://github.com/HKU-MedAI/WSI-HGNN)
