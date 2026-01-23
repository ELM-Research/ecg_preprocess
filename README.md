# ECG Preprocessing

## Installation

1. if git clone and `uv` installed, just `cd` into the repo and `uv sync`.

2. To run just do `uv run $PATH_TO_FILE`. There are examples in `scripts/`

### Base Datasets

We regard base datasets as datasets that are solely used for later mapping of external datasets. Note that `DATA_DIR` in `src/configs/constants.py` is the path to your `data` folder.

#### PTB-XL

1. Please download the PTB-XL dataset through this [link](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip).

2. Please create a `data` folder, unzip the zip file inside the `data` folder and rename the folder as `ptb_xl`.

#### MIMIC

1. Please download the Mimic IV ECG dataset through this [link](https://physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip).

2. Unzip the zip file inside the `data` directory and rename the unzipped directory as `mimic_iv`.

#### Code-15

1. First create a `code15` folder inside the `data` directory.

2. Then inside `data/code15` execute the following bash script to download the data and unzip it:

```
#!/bin/bash

for i in {0..17}; do
    echo "Downloading part ${i}..."
    wget -O "exams_part${i}.zip" "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded part ${i}"
        
        echo "Extracting part ${i}..."
        unzip -q "exams_part${i}.zip"
        
        if [ $? -eq 0 ]; then
            echo "Successfully extracted part ${i}"
            rm "exams_part${i}.zip"
        else
            echo "Error extracting part ${i}"
        fi
    else
        echo "Error downloading part ${i}"
    fi
done

echo "All downloads and extractions completed"
```

#### CSN

1. Create a `csn` folder inside the `data` directory.

2. Inside `data/csn` execute the following command in the terminal:

```
wget https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip
```

3. Unzip the file and inside of `data/csn/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0` move all of the contents outside to `data/csn`. Then you may delete the `a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0` folder.

#### CPSC

1. Create a `cpsc` folder inside the `data` directory.

2. Inside `data/cpsc` execute the following command in the terminal:

```
wget https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
```

3. Unzip the file and inside of `data/cpsc/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training` move the `cpsc_2018` and `cpsc_2018_extra` folders into the `data/cpsc` directory. Then delete the `classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2` folder.

### Mapping Datasets

Mapping datasets are datasets that are mapped to the base datasets and subsequently used for all experiments.

#### ECG-QA dataset curated by [ECG-QA, Oh et al.](https://github.com/Jwoo5/ecg-qa)

1. We exactly follow the instructions in [this section of the repository](https://github.com/Jwoo5/ecg-qa?tab=readme-ov-file#usage-notes) for mapping the PTB-XL and MIMIC IV ECG dataset to the question and answers. `cd` into ecg-qa and execute the following commands in the terminal to prepare the ECG-QA dataset.

3. To map the ECG-QA dataset to mimic and ptb, execute the following scripts respectively.

```
uv run src/datasets/map/ecg_qa/mapping_ptbxl_samples.py src/datasets/map/ecg_qa/ecgqa/ptbxl/ --ptbxl-data-dir ../data/ptb_xl
```

```
uv run src/datasets/map/ecg_qa/mapping_mimic_iv_ecg_samples.py src/datasets/map/ecg_qa/ecgqa/mimic-iv-ecg --mimic-iv-ecg-data-dir ../data/mimic
```

3. After mapping the datasets, you should have an output folder in the `data/ecg-qa` folder with the mapped `paraphrased` and `template` question and answers.

#### Pretrain MIMIC dataset curated by [ECG-Chat, Zhao et al.](https://github.com/YubaoZhao/ECG-Chat)

1. Download the `pretrain_mimic.json` file from this [dropbox link](https://www.dropbox.com/scl/fo/ccq5dxmdgg4shf02yjn8c/ANOQ1Hzj4KwHqa1b9r80uzc?rlkey=teysp3v6hg6o9uko2i4zbbjpn&e=1&st=exu3i9oo&dl=0) and place it in the corresponding folder src/datasets/map/pretrain_mimic/.

#### Instruct 45k MIMIC dataset curated by [ECG-Chat, Zhao et al.](https://github.com/YubaoZhao/ECG-Chat)

1. Download the `ecg_instruct_45k.json` file from this [link](https://github.com/YubaoZhao/ECG-Chat/blob/master/llava/playground/data/ecg_instruct_45k.json) and place it in the corresponding folder src/datasets/map/ecg_intruct_45k/.


#### ECG Instruct Pulse dataset curated by [PULSE, Liu et al.](https://github.com/AIMedLab/PULSE)

1. Downlod the `ECGInstruct.json`from this [link](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct/tree/main). Rename it to `ecg_instruct_pulse.json` and place it in the corresponding folder src/datasets/map/ecg_instruct_pulse.

#### ECG Bench Pulse dataset curated by [PULSE, Liu et al.](https://github.com/AIMedLab/PULSE)

1. The ECG Bench Pulse dataset is exclusively on HuggingFace with `.parquet` files, therefore, we utilize the `datasets` library directly to download the dataset.

#### ECG Grounding Datasets curated by [GEM, Lan et al.](https://github.com/lanxiang1017/GEM)

1. Download the `ECG_Grounding_30k.json`, `ecg-grounding-test.json` and `grounding_train_30k.json` from this [link](https://huggingface.co/datasets/LANSG/ECG-Grounding/tree/main/ecg_jsons) and place it in the corresponding folder src/datasets/map/ecg_grounding. A quick note is that `grounding_train_30k.json` is a subset of `ECG_Grounding_30k.json`, where `ECG_Grounding_30k.json` contains all 30k ECG grounding samples found in `grounding_train_30k.json`, with additional ECG conversational data from the ECG Instruct PULSE dataset.

### ECG Byte Training

We also implement training the BPE algorithm from [ECG-Byte](https://arxiv.org/abs/2412.14373). This should be trained only after preprocessing the MIMIC-IV base dataset. 
Please execute `bash scripts/train_ecg_byte.sh`.

### Hugging Face upload

We have also released the code for uploading the preprocessed, mapped datasets onto HuggingFace datasets. Please view `scripts/upload_hf.sh` for the script!