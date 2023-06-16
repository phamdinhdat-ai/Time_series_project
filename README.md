# Time_series_project



## Usage

**Setup**
- `git clone https://github.com/phamdinhdat-ai/Time_series_project`
- `cd Time_series_project`
- `pip install -r requirement.txt`

**Parameter Training**

- `--model_type` : type of network to training
- `--data_type` : type of dataset
- `--num_classes` : numbers of class in dataset 
- `--epochs`: numbers of epoch
- `--batch_size` : numbers of batch
- `--squence_lenght` : lenght of the sliding window
- `--overlap` : percent of overlap window

**Run Code** 
- `Training data with two option datasets: static and dynamic`
- For dynamic dataset: `python .\main.py --model_type mlp  --data_type dynamic --num_classes 4 --epochs 100 --batch_size 256  --sequence_lenght 10 --overlap 0.1`
- For static dataset: `python .\main.py --model_type mlp  --data_type static --num_classes 4 --epochs 100 --batch_size 256  --sequence_lenght 10 --overlap 0.1`


