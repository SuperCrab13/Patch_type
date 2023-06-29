# Patch_Type
Use hovernet pretrained on PanNuke to classify patches of WSI into 6 types (no_label, neoplastic, inflammatory, connective, necros, non_neoplastic_epithelial) <br />
Hovernet assigns nucleis in each patch, takes the most frequently predicted nucleis type as the type of corresponding patch.

## Set Up Environment
``` 
pip install -r requirement.txt
```

## Running the code
Running the following code to classify patches, considering edit the 'root_dir' argument.

``` 
python get_patches.py --root_dir YOUR_DIR
```

Related project:
- [HoverNet](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045?via%3Dihub)
- [HEAT](https://github.com/HKU-MedAI/WSI-HGNN)
