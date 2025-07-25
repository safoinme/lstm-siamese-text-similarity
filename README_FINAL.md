# LSTM Siamese Text Similarity Pipeline

A production-ready implementation of LSTM Siamese Neural Networks for text similarity matching, integrated with Kubeflow Pipelines for scalable text similarity workflows.

## 🚀 Quick Start

### Generate Pipeline YAML
```bash
python3 lstm_siamese_kubeflow_pipeline_improved.py --compile \
  --input-table preprocessed_analytics.model_reference \
  --hive-host 172.17.235.21 \
  --output lstm-siamese-pipeline.yaml
```

### Deploy to Kubeflow
1. Upload `lstm-siamese-pipeline.yaml` to Kubeflow Pipelines UI
2. Create a new run
3. Configure parameters in the UI (see Parameter Configuration section)

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Training LSTM Siamese Models](#training-lstm-siamese-models)
- [Pipeline Configuration](#pipeline-configuration)
- [Docker Images](#docker-images)
- [Parameter Configuration](#parameter-configuration)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)

## 🏗️ Architecture Overview

### Pipeline Components

The LSTM Siamese Kubeflow pipeline consists of 2 main steps:

1. **Extract Data from Hive & Run LSTM Siamese** - Queries Hive tables, converts to Siamese format, and runs similarity predictions
2. **Save Results to Hive** - Stores similarity scores and predictions back to Hive

### Data Flow

```
Hive Table → Siamese Format → LSTM Prediction → Similarity Scores → Hive Storage
     ↓             ↓              ↓               ↓              ↓
[Extract]    [Convert]      [Predict]       [Process]      [Store]
```

### Model Architecture

```
Text 1 → LSTM Encoder ↘
                       → Merge → Dense Layers → Sigmoid → Similarity Score
Text 2 → LSTM Encoder ↗
```

## 🧠 Training LSTM Siamese Models

### Using the Training Notebook

The primary way to train LSTM Siamese models is through the provided Jupyter notebook:

**`Hive_LSTM_Siamese_Training_Notebook.ipynb`**

This notebook contains:
- Complete training workflow for LSTM Siamese networks
- Data preprocessing for text pairs
- Model architecture definition
- Training with early stopping and model checkpoints
- Model evaluation and validation

### Training Steps

1. **Start Kubeflow Notebook Server**
   - Go to Kubeflow Central Dashboard → Notebooks
   - Create a new notebook server with:
     - **Image**: `172.17.232.16:9001/lstm-siamese:2.0`
     - **GPU**: Enable if available (recommended for training)
     - **CPU**: 4+ cores recommended
     - **Memory**: 16Gi+ recommended
   - Connect to the notebook server

2. **Open Training Notebook**
   - Navigate to `Hive_LSTM_Siamese_Training_Notebook.ipynb`
   - Follow the step-by-step training process
   - Adjust hyperparameters as needed

3. **Key Training Parameters**
   ```python
   # Model Configuration
   embedding_dim = 50
   max_sequence_length = 100
   number_lstm_units = 50
   number_dense_units = 50
   
   # Training Configuration
   rate_drop_lstm = 0.17
   rate_drop_dense = 0.25
   activation_function = 'relu'
   validation_split = 0.1
   
   # Task Configuration
   model_task = "text_similarity"
   ```

4. **Save Trained Model**
   - Models are saved to `/home/jovyan/models/`
   - Include model file (.h5), tokenizer (.pkl), and config (.json)
   - These models are embedded in the Docker image
   - Pipeline uses these models automatically

### Data Format for Training

Input data should be in the format:
```csv
sentences1,sentences2,is_similar
"The cat is sleeping","A cat is sleeping",1
"Dog is running","Cat is playing",0
"Good morning","Good morning everyone",1
```

### Custom Dataset Training

For your own data:

1. **Data Preparation**
   ```python
   from model import SiameseBiLSTM
   from inputHandler import word_embed_meta_data, create_test_data
   from config import siamese_config
   import pandas as pd

   # Load your data
   df = pd.read_csv('your_data.csv')
   sentences1 = list(df['sentences1'])
   sentences2 = list(df['sentences2'])
   is_similar = list(df['is_similar'])

   # Create embeddings
   tokenizer, embedding_matrix = word_embed_meta_data(
       sentences1 + sentences2, 
       siamese_config['EMBEDDING_DIM']
   )
   ```

2. **Training Script**
   ```python
   # Train the model
   siamese = SiameseBiLSTM(...)
   best_model_path = siamese.train_model(
       sentences_pair, 
       is_similar, 
       embedding_meta_data
   )
   ```

## ⚙️ Pipeline Configuration

### Main Pipeline File
**`lstm_siamese_kubeflow_pipeline_improved.py`** - Production-ready pipeline

### Key Features
- **GPU Support** - Automatic GPU detection with CPU fallback
- **Hive Integration** - Direct connection to Hive data warehouse
- **Text Preprocessing** - Automatic text cleaning and tokenization
- **Similarity Scoring** - Configurable threshold for binary classification
- **Error Handling** - Comprehensive error handling with fallback modes
- **Configurable Parameters** - All parameters configurable via Kubeflow UI

### Pipeline Generation
```bash
python3 lstm_siamese_kubeflow_pipeline_improved.py --compile \
  --input-table YOUR_TABLE_NAME \
  --hive-host YOUR_HIVE_HOST \
  --output YOUR_PIPELINE_NAME.yaml
```

### Optional Parameters
- `--no-cache` - Disable step caching
- `--input-table` - Specify input Hive table
- `--hive-host` - Hive server hostname
- `--output` - Output YAML filename

## 🐳 Docker Images

### Production Image
**`172.17.232.16:9001/lstm-siamese:2.0`**

Built from `Dockerfile.kubeflow`:
- TensorFlow 2.13.0 with GPU support
- Pre-trained word embeddings
- Hive connectivity (pyhive)
- Kubeflow Pipeline SDK
- NLTK data for text preprocessing
- Embedded models at `/home/jovyan/models/`

### Build Commands
```bash
# Build the image
docker build -f Dockerfile.kubeflow -t 172.17.232.16:9001/lstm-siamese:2.0 .

# Push to registry
docker push 172.17.232.16:9001/lstm-siamese:2.0
```

### Image Contents
```
/home/jovyan/
├── models/               # Pre-trained LSTM Siamese models
│   ├── siamese_model.h5  # Keras model
│   ├── siamese_model_tokenizer.pkl  # Tokenizer
│   └── siamese_model_config.json    # Model config
├── lstm-siamese/         # Source code
│   ├── model.py          # Model architecture
│   ├── inputHandler.py   # Data preprocessing
│   ├── config.py         # Configuration
│   └── siamese_matcher.py # Inference script
└── [other project files]
```

## 📊 Parameter Configuration

When creating a run in Kubeflow UI, configure these parameters:

### Hive Connection
```yaml
hive_host: "172.17.235.21"
hive_port: 10000
hive_user: "lhimer"
hive_database: "preprocessed_analytics"
```

### Data Parameters
```yaml
input_table: "preprocessed_analytics.model_reference"
sample_limit: null  # or integer for testing
matching_mode: "auto"  # or "production"/"testing"
```

### LSTM Siamese Model Parameters
```yaml
model_path: "/home/jovyan/models/siamese_model.h5"
max_sequence_length: 100
threshold: 0.5  # Similarity threshold for binary classification
use_gpu: false  # Set to true if GPU available
```

### Output Parameters
```yaml
save_to_hive: false  # Set to true to save results
output_table: "lstm_siamese_results"
```

### Performance Tuning
- **GPU Memory**: Set `use_gpu: false` if GPU memory issues
- **Batch Processing**: Use `sample_limit` for testing with small datasets
- **Threshold Tuning**: Adjust `threshold` based on your similarity requirements
- **Sequence Length**: Adjust `max_sequence_length` based on your text length

## 🔧 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
```yaml
# Solution: Disable GPU
use_gpu: false
```

**2. Model Loading Errors**
```bash
# Solution: Verify model files exist in Docker image
ls /home/jovyan/models/
```

**3. Hive Connection Timeout**
```yaml
# Solution: Verify Hive host and credentials
hive_host: "YOUR_CORRECT_HOST"
hive_user: "YOUR_USERNAME"
```

**4. Text Preprocessing Issues**
- Ensure text fields are not null
- Check for encoding issues in text data
- Verify tokenizer compatibility

**5. Low Similarity Scores**
- Check if model was trained on similar domain
- Adjust threshold parameter
- Verify text preprocessing steps

### Debug Mode
Enable detailed logging by checking pipeline logs in Kubeflow UI.

### Resource Requirements
- **CPU**: 4 cores minimum for inference
- **Memory**: 8Gi minimum, 16Gi recommended
- **GPU**: Optional but recommended for large models
- **Storage**: Sufficient space for models and temporary data

## 📁 File Structure

```
lstm-siamese-text-similarity/
├── README_FINAL.md                           # This file
├── lstm_siamese_kubeflow_pipeline_improved.py # 🔥 Main pipeline (PRODUCTION)
├── Hive_LSTM_Siamese_Training_Notebook.ipynb # 📚 Training notebook
├── Hive_LSTM_Siamese_Testing_Notebook.ipynb  # 🧪 Testing notebook
├── Dockerfile.kubeflow                       # 🐳 Production Docker image
├── model.py                                  # LSTM Siamese architecture
├── inputHandler.py                           # Data preprocessing
├── config.py                                 # Model configuration
├── siamese_matcher.py                        # Inference script
├── controller.py                             # Model controller
├── requirements.txt                          # Python dependencies
├── hive_siamese_data_extractor.py           # Hive data extraction
└── sample_data.csv                          # Sample training data
```

### Key Files

- **`lstm_siamese_kubeflow_pipeline_improved.py`** - Main production pipeline
- **`Hive_LSTM_Siamese_Training_Notebook.ipynb`** - Training and experimentation
- **`Dockerfile.kubeflow`** - Production Docker image definition
- **`model.py`** - LSTM Siamese neural network architecture
- **`inputHandler.py`** - Text preprocessing and data handling
- **`config.py`** - Model hyperparameters and configuration

### Deprecated Files
- `lstm_siamese_kubeflow_pipeline.py` - Original pipeline (superseded by improved version)

## 🎯 Best Practices

### Training
1. Start with the provided notebook for learning
2. Use small datasets for initial experimentation  
3. Scale up gradually with your production data
4. Save models with tokenizers and configs
5. Monitor training metrics and use early stopping

### Production Deployment
1. Test pipeline with `sample_limit` first
2. Monitor GPU memory usage if using GPU
3. Use appropriate similarity thresholds
4. Enable result saving only when needed
5. Monitor text preprocessing for edge cases

### Performance Optimization
1. Use GPU when available (`use_gpu: true`)
2. Batch process large datasets with `sample_limit`
3. Tune similarity threshold based on domain
4. Cache pipeline steps when appropriate
5. Monitor memory usage for large text datasets

## 🔬 Model Performance

### Expected Performance
- **Training Time**: 10-30 minutes on GPU for typical datasets
- **Inference Speed**: ~1000 pairs/second on GPU
- **Memory Usage**: 2-8GB depending on model size and batch size
- **Accuracy**: Depends on domain and training data quality

### Evaluation Metrics
- **Accuracy**: Binary classification accuracy
- **Precision/Recall**: For similarity detection
- **F1-Score**: Balanced measure
- **ROC-AUC**: Threshold-independent performance

## 📄 License

This project extends the original LSTM Siamese implementation for production use with Kubeflow Pipelines.

---

**Questions?** Check the training notebook or review the troubleshooting section above.