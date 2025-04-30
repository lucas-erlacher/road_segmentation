# CIL23

## Models
This is the github repositroy of our Computational Intelligence Lab Project: ETHZ CIL Road Segmentation 2023
As stated in the report we have several models that we tried and for each of them we have a seperate branch in the repository:

- Linear
- Unet
- StackedUnet
- D3+
- DoubleD3+
- DoubleD3+DiceLoss

Click [here](https://polybox.ethz.ch/index.php/s/oB6DfYf8jDml9NU) for the weights of our pre-trained models 

## Dataset
In the main repository is the code for all other parts like dataset generation scripts, etc.
To download the data either run the provided download scripts (for google authentication keys and credits are needed), otherwise you can also download our generated datasets under the following links:
- [Bing 40k dataset](https://polybox.ethz.ch/index.php/s/SEWPnvgV7vaiFZH/download)
- [Google 5k dataset](https://polybox.ethz.ch/index.php/s/7DnLxZvv2GMhdet/download)
- [Google 48k dataset](https://polybox.ethz.ch/index.php/s/tQ4kGy5vn3hebuw/download)
- [Official CIL road segmentation dataset](https://polybox.ethz.ch/index.php/s/yANTPWAzomsuCYr/download)

For a request of the full 4.3 mio bing dataset, please contact us. We only trained on the above subset of the whole bing dataset


### Bing Dataset

To generate the Bing data run the command below. Note: be aware that this probably takes very long since we didn't use paid APIs and the data is scrapped through the regular webinterface of the bing maps service. Furthermore the virtual display generates problems on some platforms. We ran this on Ubuntu 20.04.6 LTS:

```bash
python3 src/generate_bing_data.py
```

### Google Dataset

To generate the Google data insert your API key in the file and run:

```bash
python3 src/generate_google_data.py
```

## Training

In order to run a training file, head to the desired branch as listed above via (note that we assume the all libraries from the requirements.txt are present)

```bash
git checkout <branch>
```

Please make sure that all the datasets are downloaded and reside in the root directory of this repo. Furthermore the folders need to be called `google`, `bing` and `cil-road-segmentation-2022`. Then run the following command with the dataset provided as a flag (\<dataset-flag\>) from the following options (note that it depends here which of the datasets you download for the bing and google datasets since the folder names are identical):
- Bing 40k/162k dataset: -bing_data
- Bing 5k/48k dataset: -google_data
- 144 provided data samples: -official_data

In order to load a pre-trained model provide the flag `--checkpoint` (\<checkpoint-flag\>) followed by the path \<checkpoint-path\> to the model.
Then run the following command

```bash
python3 src/train.py <dataset-flag> <checkpoint-flag> <checkpoint-path>
```
Example that runs on both bing and google data:

```bash
python3 src/train.py -bing_data -google_data
```