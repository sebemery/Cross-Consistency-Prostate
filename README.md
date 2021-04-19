# Prostate_CCT

### Training

To train a model :
1) Set the config.json file with the desired parameters
2) Go to the Prostate_CCT directory and type :

```bash
python train.py --config configs/config.json --sites Name
```
Name can take different input:

```
--sites All    Call train_multi.py which automatically jointly train the six domain 
--sites BIDMC/HK/I2CVB/ISBI/ISBI_15/UCL Call train_single.py which automatically train the desired site alone
```

The log files and the `.pth` checkpoints will be saved in `saved\EXP_NAME`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```

To resume single training using a saved `.pth` model:

```bash
python train.py --config configs/config.json --resume1 saved/CCT/checkpoint.pth --sites BIDMC/HK/I2CVB/ISBI/ISBI_15/UCL
```

To resume joint training using  saved `.pth` models:

```bash
python train.py --config configs/config.json --resume1 saved/CCT/checkpoint.pth --resume2 saved/CCT/checkpoint.pth --resume3 saved/CCT/checkpoint.pth --resume4 saved/CCT/checkpoint.pth --resume5 saved/CCT/checkpoint.pth --resume6 saved/CCT/checkpoint.pth --sites All
```

The order of the saved models matter and should be : BIDMC,HK,I2CVB,ISBI,ISBI_15,UCL


**Results**: The results will be saved in `saved` as an html file, containing the validation results,
and the name it will take is `experim_name` specified in `configs/config.json`. Please `do not forget to change the name each time`.

### Inference

For inference, we need a pretrained model, the jpg images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model best_model.pth --site Name --experiment exp_name --overlay Boolean
```

The predictions will be saved as `.png` images in `outputs/site/exp_name`.

Here are the flags available for inference:

```
--site         Site name of the data to test
--model        Path to the trained pth model. (Located in /outputs/EXP_NAME/...)
--config       The config file used for training the model. (Located in /outputs/EXP_NAME/...)
--experiment   Name of the folder which will contain the results
--overlay      If True saves the mri images with an overlay of the segmentation (one for the label and one for the prediction) 
```
 
