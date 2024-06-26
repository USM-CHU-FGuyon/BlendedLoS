# Welcome the the BlendedICU code repository


## Usage:


### Setting up the BlendedICU database

The path to the BlendedICU database (version 0.3.2) should be specified in `paths.json`.

```
{
    "blendedICU": "D:/BLENDED_ICU/0.2.3/blended_data/"
}
```

Running the BlendedICU harmonization pipeline requires acess to [AmsterdamUMCdb](https://doi.org/10.17026/dans-22u-f8vd),
[eICU](https://physionet.org/content/eicu-crd/),
[HiRID](https://physionet.org/content/hirid/1.1.1/) and
[MIMIC-IV](https://physionet.org/content/mimiciv/2.2/).

For more information on the BlendedICU database, check the [ article](https://www.sciencedirect.com/science/article/pii/S153204642300223X) 
and [GitHub repo](https://github.com/USM-CHU-FGuyon/BlendedICU).


### Running BlendedICU

To facilitate reproduction of the results, scripts used for producing tables and 
figures were numbered 1 to 5.

#### 1. split_train_test

Data is partitioned using the source database (Amsterdam, eICU, HiRID 
and MIMIC-IV).
The same number of patients is extracted from each partition: source databases
are equally represented in the extracted data. 
This step prepares a training, validation and test set of each partition.

#### 2. run_experiment

This is the main part of the code.
A set of experiments can be specified in `config.json` 

The _config_ block contains parameters that are shared between all experiments. 
Parameters that are not specified in this block are given default values in 
`models/config.py`.
```
{
    "config": {
        "task":"multitask",
        "dataset": "blended",
        "use_flat": true,
        "use_med": true,
	    "use_lab": true,
        "use_vitals": true,
        "use_resp": false,
	"percentage_test": 50,
	"alpha": 20
    },
    "experiments":{
        ...
        
    }
}

```

The _experiment_ block contains parameters of each experiment. If specified
in both blocks, the _experiment_ block override parameter definition from the
_config_ block.

An _experiment_ is defined by several _runs_.
In a single run: a set of models is trained and evaluated with common parameters.
```
{
    "config": {
        ...
    },
   "experiments": {
	    "main_experiment":{
	        "models": ["tpc"], # The TPC from Rocheteau & al is used in this experiment
	        "runs":[
	             {"train_on":["mimic4"], # Training on MIMIC-IV, and keep as pretrained model
	              "percentage_trainval": 75,
	              "use_as_pretrained": true},
	             {"train_on":["amsterdam"],  # Training on AmsterdamUMC
	              "percentage_trainval": 25},
	             {"percentage_trainval_dic": {"amsterdam": 25, # Data pooling of MIMIC-IV and AmsterdamUMC
					                          "mimic4": 75}},
	             {"train_on": ["amsterdam"], # Transfer Learning from pretrained model
	              "percentage_trainval": 25,
	              "from_pretrained": true,
	              "retrain_last_layer_only": false}
	            ]
	    },
        "model_benchmark": {
	        "models": ["lstm", "transformer", "tpc"],  # Three models are run in this experiment
	        "runs":[
	            {"train_on": ["amsterdam", "hirid", "mimic4", "eicu"],
                 "percentage_trainval": 20} #There is a single run with pooled data from each source database.
                ]
        },
    }
}

```

#### 3. generate_table

Produces the results tables (including appendix) in pandas and Latex format.
The names of the experiments from step 2 should be filled in the `Experiments`
constructor.

```
self = Experiments(
    main_experiment='main_experiment',
    model_benchmark='model_benchmark',
    dataset_benchmark='dataset_benchmark',
    dataset_benchmark_nomed='dataset_benchmark_nomed',
    training_size='training_size_study',
    )
```

#### 4. extraction_table

This code produces the main table decribing the extraction of source database.




### Questions ? 

Feel free to open an Issue !

