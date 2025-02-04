## Classification Task

The code for SST-2 and AG's News datasets are in `SST2` and `AGs_News`, respectively.

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
Obtain CA
```
python generation_backdoor.py --model facebook/opt-1.3b --type clean --task negsentiment/refusal
```
Obtain ASR
```
python generation_backdoor.py --model facebook/opt-1.3b --type clean --task negsentiment/refusal
```