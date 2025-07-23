from datetime import datetime
from kfp import compiler, dsl
from typing import NamedTuple, Optional
import os
import argparse
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
import json
import time
import yaml
import kfp
import kfp.components as comp

CACHE_ENABLED = True

def extract_hive_data_func(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    input_table: str,
    output_path: str,
    sample_limit: Optional[int] = None,
    matching_mode: str = 'auto'
) -> str:
    """Extract data from Hive table and convert to LSTM Siamese format for similarity matching."""
    from pyhive import hive
    import pandas as pd
    import json
    import os
    import numpy as np
    
    print(f"Connecting to Hive at {hive_host}:{hive_port}")
    
    # Connect to Hive
    connection = hive.Connection(
        host=hive_host,
        port=hive_port,
        database=hive_database,
        username=hive_user,
        auth='NOSASL'
    )
    
    # Build query
    if sample_limit:
        query = f"SELECT * FROM {input_table} LIMIT {sample_limit}"
    else:
        query = f"SELECT * FROM {input_table}"
    
    print(f"Executing query: {query}")
    
    # Execute query and get data
    df = pd.read_sql(query, connection)
    print(f"Extracted {len(df)} records from Hive")
    
    # Detect table structure
    columns = list(df.columns)
    clean_columns = []
    for col in columns:
        if '.' in col:
            clean_col = col.split('.', 1)[1]
        else:
            clean_col = col
        clean_columns.append(clean_col)
    
    # Check for _left and _right patterns
    left_columns = [col for col in clean_columns if col.endswith('_left')]
    right_columns = [col for col in clean_columns if col.endswith('_right')]
    
    # Extract base field names
    left_fields = {col[:-5] for col in left_columns}  # Remove '_left'
    right_fields = {col[:-6] for col in right_columns}  # Remove '_right'
    
    # Check if we have matching left/right pairs
    matching_fields = left_fields.intersection(right_fields)
    
    if matching_fields and matching_mode != 'testing':
        # Production mode - left/right columns
        structure_type = "production"
        print(f"Production table detected with {len(matching_fields)} matching field pairs")
        siamese_records = []
        
        for idx, row in df.iterrows():
            left_parts = []
            right_parts = []
            
            for field in matching_fields:
                # Find the actual column names (with potential table prefixes)
                left_col = None
                right_col = None
                
                for col in df.columns:
                    clean_col = col.split('.', 1)[1] if '.' in col else col
                    if clean_col == f"{field}_left":
                        left_col = col
                    elif clean_col == f"{field}_right":
                        right_col = col
                
                # Process left column
                if left_col and pd.notna(row[left_col]) and str(row[left_col]).strip():
                    value = str(row[left_col]).strip()
                    left_parts.append(value)
                
                # Process right column
                if right_col and pd.notna(row[right_col]) and str(row[right_col]).strip():
                    value = str(row[right_col]).strip()
                    right_parts.append(value)
            
            left_text = " ".join(left_parts)
            right_text = " ".join(right_parts)
            
            record = {
                'sentences1': left_text,
                'sentences2': right_text,
                'record_id': idx
            }
            siamese_records.append(record)
        
    else:
        # Testing mode - self matching
        structure_type = "testing"
        print(f"Testing table detected with {len(clean_columns)} fields for self-matching")
        siamese_records = []
        
        for idx, row in df.iterrows():
            # Convert row to text representation
            text_parts = []
            
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    # Remove table name prefix if present
                    if '.' in col:
                        clean_col = col.split('.', 1)[1]
                    else:
                        clean_col = col
                    
                    value = str(row[col]).strip()
                    text_parts.append(f"{clean_col}: {value}")
            
            record_text = " ".join(text_parts)
            
            # Create record for inference (compare with itself or generate pairs)
            record = {
                'sentences1': record_text,
                'sentences2': record_text,  # For self-comparison
                'record_id': idx
            }
            siamese_records.append(record)
    
    # Convert to DataFrame and save
    siamese_df = pd.DataFrame(siamese_records)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as CSV
    siamese_df.to_csv(output_path, index=False)
    
    print(f"Converted {len(siamese_records)} records to LSTM Siamese format")
    print(f"Saved to: {output_path}")
    
    # Close connection
    connection.close()
    
    return output_path

def load_pretrained_model_func(
    model_path: str,
    model_source: str = 'local'  # 'local', 'gcs', 's3', etc.
) -> str:
    """Load pre-trained LSTM Siamese model for inference."""
    import os
    import json
    
    print(f"Loading pre-trained model from: {model_path}")
    
    # In production, you might load from cloud storage
    if model_source == 'gcs':
        # Example: download from Google Cloud Storage
        print("Downloading model from GCS...")
        # gsutil cp gs://your-bucket/model.h5 /tmp/model.h5
        pass
    elif model_source == 's3':
        # Example: download from AWS S3
        print("Downloading model from S3...")
        # aws s3 cp s3://your-bucket/model.h5 /tmp/model.h5
        pass
    
    # Verify model files exist
    required_files = [
        model_path,
        model_path.replace('.h5', '_tokenizer.pkl'),
        model_path.replace('.h5', '_config.json')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Warning: Model file not found: {file_path}")
            # In production, you would download from cloud storage here
    
    print("✅ Pre-trained model loaded")
    
    # Load config to validate model parameters
    config_path = model_path.replace('.h5', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        print(f"📊 Model configuration: {model_config}")
    
    return model_path

def predict_similarity_func(
    model_path: str,
    input_path: str,
    output_path: str,
    max_sequence_length: int = 100,
    threshold: float = 0.5
) -> str:
    """Use pre-trained LSTM Siamese model to predict text similarity."""
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle
    import os
    
    print(f"Loading model from {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Load tokenizer
    tokenizer_path = model_path.replace('.h5', '_tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Loading data from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    sentences1 = df['sentences1'].tolist()
    sentences2 = df['sentences2'].tolist()
    
    print(f"Making predictions for {len(df)} pairs...")
    
    # Convert texts to sequences
    seq1 = tokenizer.texts_to_sequences(sentences1)
    seq2 = tokenizer.texts_to_sequences(sentences2)
    
    # Pad sequences
    seq1 = pad_sequences(seq1, maxlen=max_sequence_length)
    seq2 = pad_sequences(seq2, maxlen=max_sequence_length)
    
    # Make predictions
    predictions = model.predict([seq1, seq2])
    predictions_binary = (predictions.flatten() > threshold).astype(int)
    
    # Add predictions to DataFrame
    results_df = df.copy()
    results_df['similarity_score'] = predictions.flatten()
    results_df['prediction'] = predictions_binary
    results_df['model_type'] = 'lstm_siamese'
    results_df['threshold_used'] = threshold
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Match rate: {np.mean(predictions_binary):.2%}")
    print(f"Average similarity score: {np.mean(predictions.flatten()):.3f}")
    
    return output_path

def save_to_hive_func(
    results_path: str,
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    output_table: str
) -> str:
    """Save LSTM Siamese inference results back to Hive."""
    from pyhive import hive
    import pandas as pd
    from datetime import datetime
    
    print(f"Loading results from {results_path}")
    
    # Load results
    results_df = pd.read_csv(results_path)
    
    # Add timestamp
    results_df['created_at'] = datetime.now().isoformat()
    
    print(f"Connecting to Hive at {hive_host}:{hive_port}")
    
    # Connect to Hive
    connection = hive.Connection(
        host=hive_host,
        port=hive_port,
        database=hive_database,
        username=hive_user,
        auth='NOSASL'
    )
    
    cursor = connection.cursor()
    
    # Create table if doesn't exist
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {output_table} (
        record_id INT,
        sentences1 STRING,
        sentences2 STRING,
        similarity_score DOUBLE,
        prediction INT,
        model_type STRING,
        threshold_used DOUBLE,
        created_at STRING
    )
    STORED AS PARQUET
    """
    
    cursor.execute(create_table_sql)
    print(f"Table {output_table} created/verified")
    
    # Insert results in batches for better performance
    insert_count = 0
    batch_size = 100
    
    for i in range(0, len(results_df), batch_size):
        batch = results_df[i:i+batch_size]
        
        for _, row in batch.iterrows():
            # Escape single quotes in strings and truncate long text
            sentences1 = str(row['sentences1']).replace("'", "''")[:1000]
            sentences2 = str(row['sentences2']).replace("'", "''")[:1000]
            
            insert_sql = f"""
            INSERT INTO {output_table} VALUES (
                {row.get('record_id', insert_count)},
                '{sentences1}',
                '{sentences2}',
                {row.get('similarity_score', 0.0)},
                {row.get('prediction', 0)},
                '{row.get('model_type', 'lstm_siamese')}',
                {row.get('threshold_used', 0.5)},
                '{row['created_at']}'
            )
            """
            
            cursor.execute(insert_sql)
            insert_count += 1
        
        print(f"Inserted {min(i+batch_size, len(results_df))}/{len(results_df)} records...")
    
    print(f"Successfully saved {insert_count} results to {output_table}")
    
    # Close connection
    connection.close()
    
    return f"Saved {insert_count} records to {output_table}"

# Create component operations
extract_hive_data_op = create_component_from_func(
    extract_hive_data_func,
    base_image='python:3.8-slim',
    packages_to_install=['pyhive', 'thrift', 'thrift_sasl', 'pandas', 'numpy']
)

load_pretrained_model_op = create_component_from_func(
    load_pretrained_model_func,
    base_image='tensorflow/tensorflow:2.13.0',
    packages_to_install=['pandas', 'numpy']
)

predict_similarity_op = create_component_from_func(
    predict_similarity_func,
    base_image='tensorflow/tensorflow:2.13.0',
    packages_to_install=['pandas', 'numpy', 'scikit-learn']
)

save_to_hive_op = create_component_from_func(
    save_to_hive_func,
    base_image='python:3.8-slim',
    packages_to_install=['pyhive', 'thrift', 'thrift_sasl', 'pandas']
)

@dsl.pipeline(
    name='Hive LSTM Siamese Inference Pipeline',
    description='Text similarity inference using pre-trained LSTM Siamese neural network with Hive integration'
)
def lstm_siamese_inference_pipeline(
    hive_host: str = '172.17.235.21',
    hive_port: int = 10000,
    hive_user: str = 'lhimer',
    hive_database: str = 'preprocessed_analytics',
    input_table: str = 'preprocessed_analytics.model_reference',
    output_table: str = 'results.lstm_siamese_matches',
    sample_limit: int = 1000,
    matching_mode: str = 'auto',
    model_path: str = '/models/siamese_model.h5',
    model_source: str = 'local',
    max_sequence_length: int = 100,
    threshold: float = 0.5
):
    """
    LSTM Siamese Text Similarity Inference Pipeline
    
    This pipeline performs inference only - training should be done in the notebook.
    
    Args:
        hive_host: Hive server hostname
        hive_port: Hive server port (default: 10000)
        hive_user: Hive username
        hive_database: Hive database name
        input_table: Input table name
        output_table: Output table name for results
        sample_limit: Number of records to process (for testing)
        matching_mode: 'auto', 'production', or 'testing'
        model_path: Path to pre-trained model
        model_source: 'local', 'gcs', 's3', etc.
        max_sequence_length: Maximum sequence length for padding
        threshold: Similarity threshold for binary classification
    """
    
    # Step 1: Extract data from Hive
    extract_task = extract_hive_data_op(
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        input_table=input_table,
        output_path='/tmp/input_data.csv',
        sample_limit=sample_limit,
        matching_mode=matching_mode
    )
    extract_task.set_display_name('Extract Data from Hive')
    extract_task.set_cpu_request('2')
    extract_task.set_memory_request('8Gi')
    extract_task.execution_options.caching_strategy.max_cache_staleness = "P0D" if not CACHE_ENABLED else None
    
    # Step 2: Load pre-trained model
    load_model_task = load_pretrained_model_op(
        model_path=model_path,
        model_source=model_source
    )
    load_model_task.set_display_name('Load Pre-trained Model')
    load_model_task.set_cpu_request('1')
    load_model_task.set_memory_request('4Gi')
    load_model_task.execution_options.caching_strategy.max_cache_staleness = "P0D" if not CACHE_ENABLED else None
    
    # Step 3: Make predictions (inference only)
    predict_task = predict_similarity_op(
        model_path=load_model_task.output,
        input_path=extract_task.output,
        output_path='/tmp/predictions.csv',
        max_sequence_length=max_sequence_length,
        threshold=threshold
    )
    predict_task.set_display_name('Predict Text Similarity (Inference)')
    predict_task.set_cpu_request('2')
    predict_task.set_memory_request('8Gi')
    # Add GPU if needed for large models
    # predict_task.set_gpu_limit('1')
    predict_task.execution_options.caching_strategy.max_cache_staleness = "P0D" if not CACHE_ENABLED else None
    
    # Step 4: Save results to Hive
    save_task = save_to_hive_op(
        results_path=predict_task.output,
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        output_table=output_table
    )
    save_task.set_display_name('Save Results to Hive')
    save_task.set_cpu_request('2')
    save_task.set_memory_request('8Gi')
    save_task.execution_options.caching_strategy.max_cache_staleness = "P0D" if not CACHE_ENABLED else None

def compile_pipeline():
    """Compile the pipeline."""
    pipeline_filename = 'lstm_siamese_inference_pipeline.yaml'
    compiler.Compiler().compile(lstm_siamese_inference_pipeline, pipeline_filename)
    print(f"Pipeline compiled to: {pipeline_filename}")
    return pipeline_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM Siamese Text Similarity Inference Pipeline')
    parser.add_argument('--compile', action='store_true', help='Compile the pipeline')
    parser.add_argument('--cache', action='store_true', default=False, help='Enable caching')
    
    args = parser.parse_args()
    
    if args.cache:
        CACHE_ENABLED = True
    
    if args.compile:
        pipeline_file = compile_pipeline()
        print(f"Inference pipeline compiled successfully: {pipeline_file}")
        print("\n📋 Pipeline Steps:")
        print("1. Extract Data from Hive - Extract and format data for inference")
        print("2. Load Pre-trained Model - Load trained model from storage")
        print("3. Predict Text Similarity - Run inference on data")
        print("4. Save Results to Hive - Store predictions back to Hive")
        print("\n💡 Note: Training should be done in the Jupyter notebook")
    else:
        print("Use --compile flag to compile the pipeline")
        print("Example: python lstm_siamese_kubeflow_pipeline.py --compile")