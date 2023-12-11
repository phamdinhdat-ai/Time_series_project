
### Installation
- `git clone https://github.com/phamdinhdat-ai/Time_series_project`
- `cd Time_series_project`
- `git checkout Jounar_2023`
- `pip install -r requirement.txt`

### Run

**Comet ML API Key**

Firstly, you need to create an account in [Comet ML](https://www.comet.com/site). Then you get your api key and put it in the scripts in `trainer.py` as  below:  
```python
  from comet_ml import Experiment
    experiment = Experiment(
    api_key="********your key***", # adđ your comet api key in here. 
    project_name="journal-2023",
    workspace="datphamai"
)
```

- with Scenario 1:

  - Our lstm model:
  `python main.py --model_type lstm --data_type static --scenario sample_divide --num_classes 12 --epochs 10 --sequence_length 20 --overlap 0.8 --batch_size 512 --normalizer batch_norm --loss_fn nll`
  

- with Scenario 2:

  - Baseline model:
  `python main.py --model_type lstm --data_type static --num_classes 12 --epochs 10 --sequence_length 20 --overlap 0.8 --batch_size 512 --normalizer batch_norm --loss_fn nll`
