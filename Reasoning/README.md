## Reasoning Task

### Confidence Selection
Obtain the highest confident examples
```
python confidence_selection.py
```

### Similarity Selection
Obtain the highest similar examples
```
python similarity_selection.py
```

### Defensive Demonstration
Combining examples of defences searched for from confident and similarity choices with poisoning demonstrations.

### Getting Results
Obtain all results by
```
python3 cot_eval.py --exp_name $exp_name --exp_path $exp_path --exp_config $exp_config
```

- exp_name=`gsm8k` (replace with the filename of *.yaml)
- exp_path=`experiments/*` (where the experiment results are stored)
- exp_config=`configs/*` (path to the *.yaml file, e.g. configs/cot/badchain/llama3_70b)