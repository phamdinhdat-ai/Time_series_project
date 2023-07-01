# Time_series_project



## Usage

**Setup**
- `git clone https://github.com/phamdinhdat-ai/Time_series_project`
- `cd Time_series_project`
- `git checkout SPP_2023_v1`
- `pip install -r requirement.txt`


**Data**
- In `/data` folder, create 4 sub-folder: 
- `static`: add static dataset to this folder 
- `dynamic`: add dynamic dataset to this folder 
- `5_fold_dynamic`: add k-fold dynamic dataset 
- `5_fold_static`: add k-fold static dataset


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

- For k-fold dynamic dataset: `!python main_k_fold.py --model_type mlp --data_type k_fold_dynamic --num_classes 4 --epochs 100 --batch_size 512 --sequence_lenght 10 --overlap 0.9`
- For k-fold static dataset: `!python main_k_fold.py --model_type mlp --data_type k_fold_static --num_classes 13 --epochs 100 --batch_size 512 --sequence_lenght 10 --overlap 0.9`
