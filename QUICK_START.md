# LSTM Siamese Text Similarity Quick Start

## 🚀 Generate Pipeline YAML (Working Command)

```bash
python3 lstm_siamese_kubeflow_pipeline_improved.py --compile \
  --input-table preprocessed_analytics.model_reference \
  --hive-host 172.17.235.21 \
  --output lstm-siamese-pipeline.yaml
```

## 📊 Kubeflow UI Parameters

When starting a run in Kubeflow UI, set these parameters:

### Required Parameters
- **hive_host**: `172.17.235.21`
- **input_table**: `preprocessed_analytics.model_reference` 
- **use_gpu**: `false` (or `true` if GPU available)

### Optional Parameters  
- **sample_limit**: Leave empty for full dataset, or set integer for testing
- **model_path**: `/home/jovyan/models/siamese_model.h5` (default)
- **max_sequence_length**: `100` (adjust based on your text length)
- **threshold**: `0.5` (similarity threshold for binary classification)
- **save_to_hive**: `false` (set to `true` to save results)
- **output_table**: `lstm_siamese_results` (if saving to Hive)

## 🐳 Docker Image
Current production image: `172.17.232.16:9001/lstm-siamese:2.0`

## 📚 Training
1. Create Kubeflow Notebook Server with image: `172.17.232.16:9001/lstm-siamese:2.0`
2. Open `Hive_LSTM_Siamese_Training_Notebook.ipynb` in the notebook server
3. Follow the training steps in the notebook

## 🔬 Model Performance
- **Accuracy**: Depends on training data and domain
- **Speed**: ~1000 text pairs/second on GPU
- **Memory**: 2-8GB depending on model size

## ⚠️ Troubleshooting
- GPU issues → Set `use_gpu: false`
- Model loading errors → Check `/home/jovyan/models/` directory
- Low similarity scores → Adjust `threshold` parameter
- Text preprocessing issues → Verify input data format