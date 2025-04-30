style:
	black . --exclude dataset_40K/ --exclude cil-road-segmentation-2022

check-style:
	black . --exclude dataset_40K/ --exclude cil-road-segmentation-2022 --check

download-data: download-official-dataset

download-official-dataset:
	wget "https://polybox.ethz.ch/index.php/s/yANTPWAzomsuCYr/download" -O "dataset.zip"
	unzip dataset
	rm dataset.zip