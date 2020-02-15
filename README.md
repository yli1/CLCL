# Compositional Language Continual Learning

Li, Y., Zhao, L., Church, K., and Elhoseiny, M. Compositional language continual learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rklnDgHtDS.

### Dependencies
The code needs TensorFlow, and the following requirments. 
```buildoutcfg
pip install -r requirements.txt
sudo apt-get install python-tk
```
### Instruction Learning Experiments
```buildoutcfg
# Please fix CUDA_VISIBLE_DEVICES in the scrips.
python sequential_data_generator.py --data_name data_scan
sh experiments_scan/main_proposed_scan/main_proposed_scan.a.sh
sh experiments_scan/baseline_Standard_scan/baseline_Standard_scan.a.sh
sh experiments_scan/baseline_Compositional_scan/baseline_Compositoinal_scan.a.sh
sh experiments_scan/baseline_EWC_scan/baseline_EWC_scan.a.sh
sh experiments_scan/baseline_MAS_scan/baseline_MAS_scan.a.sh

# Plotting results
python scan_evalutation_plot.py
```

### Machine Translation Experiments
```buildoutcfg
# Please fix CUDA_VISIBLE_DEVICES in the scrips.
python sequential_data_generator.py --data_name data_translate
sh experiments_translate/main_proposed_translate/main_proposed_translate.a.sh
sh experiments_translate/baseline_Standard_translate/baseline_Standard_translate.a.sh
sh experiments_translate/baseline_Compositional_translate/baseline_Compositoinal_translate.a.sh
sh experiments_translate/baseline_EWC_translate/baseline_EWC_translate.a.sh
sh experiments_translate/baseline_MAS_translate/baseline_MAS_translate.a.sh

# Plotting results
python machine_translation_evalutation_plot.py
```

### Visualization
```buildoutcfg
# Visualize attention maps
sh continual_learning_attension_visualization.sh

# Visualize embeddings
sh continual_learning_embedding_visualization.sh
```