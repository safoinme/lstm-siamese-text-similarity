# Hive-LSTM Siamese Text Similarity Integration

This project provides a complete pipeline for text similarity matching using LSTM Siamese neural networks with Apache Hive integration and Kubeflow deployment capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Apache Hive cluster
- Kubeflow (optional, for production deployment)
- GPU support (recommended for training)

### Installation

1. Clone/download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 📊 Data Format

The system supports two data formats:

### Production Format (Left/Right columns)
```csv
id,firstname_left,lastname_left,firstname_right,lastname_right,similarity
1,John,Smith,Jon,Smith,1
2,Mary,Johnson,Maria,Gonzalez,0
```

### Testing Format (Regular columns)
```csv
id,firstname,lastname,email
1,John,Smith,john.smith@email.com
2,Mary,Johnson,mary.j@email.com
```

## 🔧 Usage

### 1. Testing with Jupyter Notebook

Use `Hive_LSTM_Siamese_Testing_Notebook.ipynb` for interactive testing and development:

```bash
jupyter notebook Hive_LSTM_Siamese_Testing_Notebook.ipynb
```

The notebook provides:
- Hive connection testing
- Data extraction and conversion
- Model training and evaluation
- Result analysis and visualization
- Kubeflow pipeline configuration generation

### 2. Command Line Tools

#### Extract Data from Hive

```bash
python hive_siamese_data_extractor.py \
  --host YOUR_HIVE_HOST \
  --port 10000 \
  --username YOUR_USERNAME \
  --database YOUR_DATABASE \
  --table YOUR_TABLE \
  --output data/extracted_data.csv \
  --limit 1000 \
  --mode auto \
  --balance
```

#### Train and Predict with LSTM Siamese

```bash
python siamese_matcher.py \
  --input_path data/extracted_data.csv \
  --output_path results/predictions.csv \
  --model_path models/siamese_model.h5 \
  --mode train_predict \
  --epochs 20 \
  --batch_size 64
```

### 3. Configuration Options

#### Matching Modes

- `auto`: Automatically detect table structure
- `production`: Force left/right column matching
- `testing`: Force self-matching mode
- `cross_product`: Generate all possible pairs (for training data creation)

#### Model Parameters

```json
{
  "embedding_dim": 300,
  "max_sequence_length": 100,
  "number_lstm": 50,
  "rate_drop_lstm": 0.25,
  "number_dense_units": 50,
  "activation_function": "relu",
  "rate_drop_dense": 0.25,
  "validation_split": 0.2,
  "epochs": 10,
  "batch_size": 64
}
```

## 🏗️ Architecture

### LSTM Siamese Neural Network

```
Input 1 (Text) ──► Embedding ──► Bi-LSTM ──┐
                                          ├── Distance ──► Dense ──► Sigmoid ──► Similarity Score
Input 2 (Text) ──► Embedding ──► Bi-LSTM ──┘
```

### Components

1. **Data Extractor** (`hive_siamese_data_extractor.py`): Extract and convert Hive data
2. **Siamese Matcher** (`siamese_matcher.py`): Train and predict with LSTM Siamese
3. **Notebook** (`Hive_LSTM_Siamese_Testing_Notebook.ipynb`): Interactive testing
4. **Kubeflow Pipeline** (`lstm_siamese_kubeflow_pipeline.py`): Production deployment

## 🚀 Kubeflow Deployment

### 1. Build Docker Image

```bash
docker build -t your-registry/lstm-siamese-hive:latest -f Dockerfile .
docker push your-registry/lstm-siamese-hive:latest
```

### 2. Compile Pipeline

```bash
python lstm_siamese_kubeflow_pipeline.py --compile
```

### 3. Deploy to Kubeflow

1. Upload the compiled YAML to Kubeflow Pipelines UI
2. Create a new run with your parameters
3. Monitor execution and results

### Pipeline Steps

1. **Extract Data**: Pull data from Hive table
2. **Train Model**: Train LSTM Siamese on extracted data
3. **Predict**: Make similarity predictions
4. **Save Results**: Store results back to Hive

## 📊 Performance Tuning

### For Large Datasets

- Use `--limit` parameter to test with smaller samples first
- Increase `batch_size` for faster training (if memory allows)
- Use GPU acceleration for training
- Consider data balancing for skewed datasets

### Memory Optimization

- Reduce `max_sequence_length` for shorter texts
- Lower `embedding_dim` if needed
- Use gradient checkpointing for very large models

### Training Tips

- Start with fewer epochs (5-10) for initial testing
- Use validation split to monitor overfitting
- Balance similar/dissimilar pairs in training data
- Consider early stopping for long training runs

## 🔍 Monitoring and Results

### Result Analysis

The system provides detailed analytics:

- Similarity score distribution
- Match/non-match ratios
- Confusion matrices (when ground truth available)
- Training curves and metrics

### Output Format

Results are saved as CSV with columns:
- `sentences1`: First text in pair
- `sentences2`: Second text in pair
- `similarity_score`: Raw similarity score (0-1)
- `prediction`: Binary prediction (0/1)
- `ground_truth`: Actual label (if available)
- `timestamp`: Processing timestamp

## 🐛 Troubleshooting

### Common Issues

1. **Connection Errors**
   ```
   Failed to connect to Hive
   ```
   - Check Hive host/port settings
   - Verify network connectivity
   - Check authentication credentials

2. **Memory Errors**
   ```
   OutOfMemoryError during training
   ```
   - Reduce batch_size
   - Lower sequence length
   - Use smaller embedding dimension

3. **Performance Issues**
   ```
   Training very slow
   ```
   - Enable GPU acceleration
   - Increase batch_size
   - Reduce dataset size for testing

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Comparison with DITTO

| Aspect | LSTM Siamese | DITTO |
|--------|-------------|-------|
| Model Type | Bidirectional LSTM | BERT-based Transformer |
| Training Time | Faster | Slower |
| Memory Usage | Lower | Higher |
| Accuracy | Good | Excellent |
| Interpretability | Moderate | Low |
| Custom Training | Easy | Complex |

### When to Use LSTM Siamese

- Limited computational resources
- Need for faster inference
- Custom domain-specific training
- Interpretable similarity scores
- Streaming/real-time processing

### When to Use DITTO

- Maximum accuracy is critical
- Large amounts of training data
- State-of-the-art performance needed
- Pre-trained knowledge is valuable

## 🤝 Contributing

1. Test changes with the Jupyter notebook
2. Ensure all components work together
3. Update documentation as needed
4. Test Kubeflow pipeline compilation

## 📚 References

1. [Siamese Recurrent Architectures for Learning Sentence Similarity](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)
2. [Deep LSTM Siamese Network for Text Similarity](https://github.com/dhwajraj/deep-siamese-text-similarity)
3. [Apache Hive Documentation](https://hive.apache.org/)
4. [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
5. [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

## 📄 License

This project uses the same license as the original LSTM Siamese implementation.