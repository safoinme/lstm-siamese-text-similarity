from datetime import datetime
from kfp import compiler, dsl
from typing import Optional
import os
import argparse
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
import json
import time

CACHE_ENABLED = True

def safe_table_name(table: str) -> str:
    """Return the table part of a fully-qualified hive table name."""
    return table.split('.')[-1]

def extract_hive_data_func(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    input_table: str,
    output_path: str,
    sample_limit: int = 0,
    matching_mode: str = 'auto'
) -> str:
    """Extract data from Hive table and convert to LSTM Siamese format."""
    from pyhive import hive
    import pandas as pd
    import json
    import os
    from datetime import datetime
    
    # Setup logging to shared volume
    log_dir = "/data/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/extract_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log_and_print(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    log_and_print("=== EXTRACT STARTED ===")
    log_and_print(f"Hive Host: {hive_host}:{hive_port}")
    log_and_print(f"Database: {hive_database}")
    log_and_print(f"Input Table: {input_table}")
    log_and_print(f"Output Path: {output_path}")
    log_and_print(f"Sample Limit: {sample_limit}")
    log_and_print(f"Matching Mode: {matching_mode}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Connect to Hive
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            username=hive_user,
            database=hive_database
        )
        
        # Extract data
        query = f"SELECT * FROM {input_table}"
        if sample_limit and sample_limit > 0:
            query += f" LIMIT {sample_limit}"
        
        df = pd.read_sql(query, connection)
        print(f"Extracted {len(df)} records from {input_table}")
        print(f"Columns: {list(df.columns)}")
        
        # Detect and convert to LSTM Siamese format
        def detect_table_structure(df):
            columns = list(df.columns)
            clean_columns = []
            for col in columns:
                if '.' in col:
                    clean_col = col.split('.', 1)[1]
                else:
                    clean_col = col
                clean_columns.append(clean_col)
            
            left_columns = [col for col in clean_columns if col.endswith('_left')]
            right_columns = [col for col in clean_columns if col.endswith('_right')]
            left_fields = {col[:-5] for col in left_columns}
            right_fields = {col[:-6] for col in right_columns}
            matching_fields = left_fields.intersection(right_fields)
            
            if matching_fields:
                structure_type = "production"
                message = f"Production table detected with {len(matching_fields)} matching field pairs"
            else:
                structure_type = "testing"
                message = f"Testing table detected with {len(clean_columns)} fields for self-matching"
            
            return {
                'type': structure_type,
                'columns': columns,
                'clean_columns': clean_columns,
                'matching_fields': list(matching_fields),
                'message': message
            }
        
        def convert_production_format(df, structure):
            records = []
            for idx, row in df.iterrows():
                left_parts = []
                right_parts = []
                
                for field in structure['matching_fields']:
                    left_col = None
                    right_col = None
                    
                    for col in df.columns:
                        clean_col = col.split('.', 1)[1] if '.' in col else col
                        if clean_col == f"{field}_left":
                            left_col = col
                        elif clean_col == f"{field}_right":
                            right_col = col
                    
                    if left_col and pd.notna(row[left_col]) and str(row[left_col]).strip():
                        value = str(row[left_col]).strip()
                        left_parts.append(value)
                    
                    if right_col and pd.notna(row[right_col]) and str(row[right_col]).strip():
                        value = str(row[right_col]).strip()
                        right_parts.append(value)
                
                left_text = " ".join(left_parts)
                right_text = " ".join(right_parts)
                
                record = {
                    "sentence1": left_text,
                    "sentence2": right_text,
                    "id": idx
                }
                records.append(record)
            
            return records
        
        def convert_testing_format(df, structure):
            records = []
            for idx, row in df.iterrows():
                col_val_parts = []
                
                for col in df.columns:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        if '.' in col:
                            clean_col = col.split('.', 1)[1]
                        else:
                            clean_col = col
                        
                        value = str(row[col]).strip()
                        col_val_parts.append(value)
                
                record_text = " ".join(col_val_parts)
                
                record = {
                    "sentence1": record_text,
                    "sentence2": record_text,
                    "id": idx
                }
                records.append(record)
            
            return records
        
        def convert_to_siamese_format(df, matching_mode):
            structure = detect_table_structure(df)
            
            # Override detection if mode is specified
            if matching_mode == 'production':
                structure['type'] = 'production'
            elif matching_mode == 'testing':
                structure['type'] = 'testing'
            
            print(structure['message'])
            
            if structure['type'] == 'production':
                if not structure['matching_fields']:
                    raise ValueError("Production mode requires _left/_right column pairs, but none were found!")
                return convert_production_format(df, structure)
            else:
                return convert_testing_format(df, structure)
        
        siamese_records = convert_to_siamese_format(df, matching_mode)
        print(f"Converted {len(siamese_records)} records to LSTM Siamese format")
        
        # Save to JSONL file in the format expected by the similarity model
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in siamese_records:
                f.write(json.dumps(record, ensure_ascii=True) + '\n')
        
        log_and_print(f"Saved to: {output_path}")
        
        connection.close()
        log_and_print("=== EXTRACT COMPLETE ===")
        return output_path
        
    except Exception as e:
        log_and_print(f"ERROR: {str(e)}")
        log_and_print("=== EXTRACT FAILED ===")
        raise

def run_lstm_siamese_func(
    input_path: str,
    output_path: str,
    model_path: str,
    tokenizer_path: str,
    max_sequence_length: int,
    batch_size: int,
    similarity_threshold: float,
    use_gpu: bool
) -> str:
    """Run LSTM Siamese matching."""
    import json
    import os
    import numpy as np
    from collections import namedtuple
    from datetime import datetime
    
    # Setup logging to shared volume
    log_dir = "/data/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/lstm_siamese_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log_and_print(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    log_and_print("=== MATCHING STARTED ===")
    log_and_print(f"Input Path: {input_path}")
    log_and_print(f"Output Path: {output_path}")
    log_and_print(f"Model Path: {model_path}")
    log_and_print(f"Max Sequence Length: {max_sequence_length}")
    log_and_print(f"Batch Size: {batch_size}")
    log_and_print(f"Similarity Threshold: {similarity_threshold}")
    log_and_print(f"Use GPU: {use_gpu}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # GPU Detection and Setup
        gpu_available = False
        if use_gpu:
            log_and_print("=== GPU SETUP ===")
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    log_and_print(f"GPU Available: {len(gpus)} GPU(s) found")
                    # Configure GPU memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    gpu_available = True
                else:
                    log_and_print("WARNING: GPU requested but not available, falling back to CPU")
            except Exception as e:
                log_and_print(f"WARNING: TensorFlow GPU setup failed: {str(e)}, falling back to CPU")
        else:
            log_and_print("=== CPU MODE ===")
        
        # Load input data
        log_and_print("Loading input data...")
        records = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        log_and_print(f"Loaded {len(records)} text pairs for similarity matching")
        
        # Load model and tokenizer
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            import pickle
            
            log_and_print(f"Loading LSTM Siamese model from {model_path}")
            model = load_model(model_path)
            
            log_and_print(f"Loading tokenizer from {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Prepare data for prediction
            sentence1_list = [record['sentence1'] for record in records]
            sentence2_list = [record['sentence2'] for record in records]
            
            log_and_print(f"Converting texts to sequences...")
            
            # Convert texts to sequences
            seq1 = tokenizer.texts_to_sequences(sentence1_list)
            seq2 = tokenizer.texts_to_sequences(sentence2_list)
            
            # Pad sequences
            seq1 = pad_sequences(seq1, maxlen=max_sequence_length)
            seq2 = pad_sequences(seq2, maxlen=max_sequence_length)
            
            # Create leak features (simple features like length difference)
            leaks = []
            for s1, s2 in zip(sentence1_list, sentence2_list):
                leak_features = [
                    len(s1), len(s2),
                    abs(len(s1) - len(s2)),
                    len(set(s1.split()).intersection(set(s2.split())))
                ]
                leaks.append(leak_features)
            
            leaks = np.array(leaks)
            
            log_and_print(f"Making predictions for {len(records)} text pairs...")
            
            # Make predictions in batches
            predictions = model.predict([seq1, seq2, leaks], batch_size=batch_size)
            predictions_binary = (predictions.flatten() > similarity_threshold).astype(int)
            
            # Process results
            results = []
            metrics = {"total_pairs": 0, "matches": 0, "non_matches": 0, "avg_similarity": 0.0}
            
            for i, record in enumerate(records):
                similarity_score = float(predictions[i][0])
                is_match = int(predictions_binary[i])
                
                result_data = {
                    'id': record.get('id', i),
                    'sentence1': record['sentence1'],
                    'sentence2': record['sentence2'],
                    'similarity_score': similarity_score,
                    'match': is_match,
                    'match_confidence': similarity_score,
                    'model_type': 'lstm_siamese',
                    'threshold_used': similarity_threshold
                }
                results.append(result_data)
                
                metrics["total_pairs"] += 1
                if is_match == 1:
                    metrics["matches"] += 1
                else:
                    metrics["non_matches"] += 1
            
            metrics["avg_similarity"] = float(np.mean(predictions.flatten()))
            
            # Save results in JSON Lines format
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=True) + '\n')
            
            log_and_print(f"Matching completed. Metrics: {metrics}")
            log_and_print(f"Match rate: {predictions_binary.mean():.2%}")
            log_and_print(f"Average similarity score: {metrics['avg_similarity']:.3f}")
            log_and_print(f"GPU used: {gpu_available}")
            log_and_print("=== MATCHING COMPLETE ===")
            
            return output_path
            
        except Exception as e:
            log_and_print(f"ERROR: {str(e)}")
            log_and_print("Falling back to simple text matching...")
            
            # Fallback: simple exact text matching
            results = []
            metrics = {"total_pairs": 0, "matches": 0, "non_matches": 0, "avg_similarity": 0.0}
            
            for i, record in enumerate(records):
                # Simple exact match fallback
                s1 = record['sentence1'].lower().strip()
                s2 = record['sentence2'].lower().strip()
                is_exact_match = s1 == s2
                similarity_score = 1.0 if is_exact_match else 0.0
                
                result_data = {
                    'id': record.get('id', i),
                    'sentence1': record['sentence1'],
                    'sentence2': record['sentence2'],
                    'similarity_score': similarity_score,
                    'match': 1 if is_exact_match else 0,
                    'match_confidence': similarity_score,
                    'model_type': 'fallback_exact_match',
                    'threshold_used': similarity_threshold
                }
                results.append(result_data)
                
                metrics["total_pairs"] += 1
                if is_exact_match:
                    metrics["matches"] += 1
                else:
                    metrics["non_matches"] += 1
            
            if metrics["total_pairs"] > 0:
                metrics["avg_similarity"] = metrics["matches"] / metrics["total_pairs"]
            
            # Save fallback results
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=True) + '\n')
            
            log_and_print(f"Fallback matching completed. Metrics: {metrics}")
            log_and_print("=== MATCHING COMPLETE (FALLBACK) ===")
            
            return output_path
        
    except Exception as e:
        log_and_print(f"ERROR: {str(e)}")
        log_and_print("=== MATCHING FAILED ===")
        raise

def save_results_to_hive_func(
    results_path: str,
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    output_table: str,
    save_results: bool
) -> str:
    """Save matching results to Hive."""
    if not save_results:
        print("Skipping Hive save as save_results is False")
        return "Skipped"
    
    from pyhive import hive
    import pandas as pd
    import json
    import tempfile
    import os
    
    try:
        # Read results
        results = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                results.append({
                    'record_id': data.get('id', 0),
                    'sentence1': str(data.get('sentence1', ''))[:1000],  # Truncate long text
                    'sentence2': str(data.get('sentence2', ''))[:1000],
                    'similarity_score': data.get('similarity_score', 0.0),
                    'match': data.get('match', 0),
                    'match_confidence': data.get('match_confidence', 0.0),
                    'model_type': data.get('model_type', 'lstm_siamese'),
                    'threshold_used': data.get('threshold_used', 0.5),
                    'processing_timestamp': datetime.now().isoformat()
                })
        
        if not results:
            print("No results to save")
            return "No results"
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Connect to Hive
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            username=hive_user,
            database=hive_database
        )
        
        cursor = connection.cursor()
        
        # Create table if it doesn't exist
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            record_id INT,
            sentence1 STRING,
            sentence2 STRING,
            similarity_score DOUBLE,
            match INT,
            match_confidence DOUBLE,
            model_type STRING,
            threshold_used DOUBLE,
            processing_timestamp STRING
        )
        STORED AS PARQUET
        """
        
        cursor.execute(create_table_sql)
        
        # Save DataFrame to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False, header=False)
            temp_path = temp_file.name
        
        try:
            # Load data into Hive table
            load_sql = f"""
            LOAD DATA LOCAL INPATH '{temp_path}' 
            INTO TABLE {output_table}
            """
            cursor.execute(load_sql)
            
            print(f"Successfully saved {len(results)} results to {output_table}")
            return f"Saved {len(results)} results to {output_table}"
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        connection.close()
        
    except Exception as e:
        print(f"Error saving to Hive: {str(e)}")
        return f"Error: {str(e)}"

def create_log_summary_func() -> str:
    """Create log summary."""
    import os
    import glob
    from datetime import datetime
    
    log_dir = "/data/logs"
    summary_file = f"{log_dir}/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    try:
        log_files = glob.glob(f"{log_dir}/*.log")
        log_files.sort()
        
        with open(summary_file, 'w') as summary:
            summary.write(f"=== PIPELINE SUMMARY ===\n")
            summary.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary.write(f"Total log files: {len(log_files)}\n\n")
            
            for log_file in log_files:
                if log_file != summary_file:
                    summary.write(f"\n{'='*50}\n")
                    summary.write(f"LOG FILE: {os.path.basename(log_file)}\n")
                    summary.write(f"{'='*50}\n")
                    
                    try:
                        with open(log_file, 'r') as f:
                            summary.write(f.read())
                    except Exception as e:
                        summary.write(f"Error reading {log_file}: {str(e)}\n")
            
            summary.write(f"\n{'='*50}\n")
            summary.write("=== END SUMMARY ===\n")
        
        print(f"Log summary created: {summary_file}")
        return summary_file
        
    except Exception as e:
        print(f"Error creating log summary: {str(e)}")
        return f"Error: {str(e)}"

# Create Kubeflow components
extract_hive_data_op = create_component_from_func(
    func=extract_hive_data_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',
)

run_lstm_siamese_op = create_component_from_func(
    func=run_lstm_siamese_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',
)

save_results_to_hive_op = create_component_from_func(
    func=save_results_to_hive_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',
)

create_log_summary_op = create_component_from_func(
    func=create_log_summary_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',
)

def generate_pipeline_name(input_table: str) -> str:
    """Generate a unique pipeline name based on table and timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_safe = safe_table_name(input_table).replace('_', '-')
    return f"siamese-{table_safe}-{timestamp}"

@dsl.pipeline(
    name="siamese",
    description="Siamese Pipeline"
)
def siamese_pipeline(
    # Hive connection parameters
    hive_host: str = "172.17.235.21",
    hive_port: int = 10000,
    hive_user: str = "lhimer",
    hive_database: str = "preprocessed_analytics",
    
    # Input table
    input_table: str = "preprocessed_analytics.model_reference",
    
    # Data limits (for testing)
    sample_limit: int = 0,
    
    # Matching mode
    matching_mode: str = 'auto',
    
    # LSTM Siamese model parameters
    model_path: str = "/home/jovyan/models/lstm_siamese_model.h5",
    tokenizer_path: str = "/home/jovyan/models/tokenizer.pkl",
    max_sequence_length: int = 100,
    batch_size: int = 32,
    similarity_threshold: float = 0.5,
    use_gpu: bool = False,
    
    # Output parameters
    save_to_hive: bool = False,
    output_table: str = "lstm_siamese_results"
):
    """LSTM Siamese text similarity pipeline."""
    
    # Define environment variables for Hive connectivity
    env_vars = [
        V1EnvVar(name='HIVE_HOST', value=hive_host),
        V1EnvVar(name='HIVE_PORT', value=str(hive_port)),
        V1EnvVar(name='HIVE_USER', value=hive_user),
        V1EnvVar(name='HIVE_DATABASE', value=hive_database)
    ]
    
    # Create a new PVC for this pipeline run
    from kubernetes import client as k8s_client
    
    # Create a new PVC dynamically
    vop = dsl.VolumeOp(
        name="create-siamese-pvc",
        resource_name="siamese-data-pvc",
        size="10Gi",
        modes=["ReadWriteOnce"]
    ).volume
    
    # Step 1: Extract data from Hive table and create pairs
    extract_data = extract_hive_data_op(
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        input_table=input_table,
        output_path="/data/input/pairs.jsonl",
        sample_limit=sample_limit,
        matching_mode=matching_mode
    )
    
    # Add volume and environment variables
    extract_data.add_pvolumes({'/data': vop})
    for env_var in env_vars:
        extract_data.add_env_variable(env_var)
    extract_data.set_display_name('Extract Data')
    extract_data.set_caching_options(enable_caching=CACHE_ENABLED)
    
    # Step 2: Run LSTM Siamese matching
    matching_results = run_lstm_siamese_op(
        input_path="/data/input/pairs.jsonl",
        output_path="/data/output/similarity_results.jsonl",
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        similarity_threshold=similarity_threshold,
        use_gpu=use_gpu
    ).after(extract_data)
    
    # Add volume and GPU resources
    matching_results.add_pvolumes({'/data': vop})
    matching_results.set_display_name('Siamese Match')  
    
    if use_gpu:
        matching_results.set_gpu_limit(1)
        matching_results.set_memory_limit('16Gi')
        matching_results.set_memory_request('8Gi')
    else:
        matching_results.set_memory_limit('8Gi')
        matching_results.set_memory_request('4Gi')
        
    matching_results.set_cpu_limit('4')
    matching_results.set_cpu_request('2')
    matching_results.set_caching_options(enable_caching=False)  # Don't cache matching results
    
    # Step 3: Optionally save results to Hive
    save_results = save_results_to_hive_op(
        results_path="/data/output/similarity_results.jsonl",
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        output_table=output_table,
        save_results=save_to_hive
    ).after(matching_results)
    
    # Add volume and environment variables
    save_results.add_pvolumes({'/data': vop})
    for env_var in env_vars:
        save_results.add_env_variable(env_var)
    save_results.set_display_name('Save Results')
    save_results.set_caching_options(enable_caching=False)
    
    # Step 4: Create log summary (always runs last)
    log_summary = create_log_summary_op().after(save_results)
    log_summary.add_pvolumes({'/data': vop})
    log_summary.set_display_name('Logs')
    log_summary.set_caching_options(enable_caching=False)

def compile_pipeline(
    input_table: str = "preprocessed_analytics.model_reference",
    hive_host: str = "172.17.235.21",
    pipeline_file: str = "lstm-siamese-pipeline.yaml"
):
    """Compile pipeline."""
    try:
        compiler.Compiler().compile(
            pipeline_func=siamese_pipeline,
            package_path=pipeline_file,
            type_check=True
        )
        
        pipeline_name = generate_pipeline_name(input_table)
        print(f"\nPipeline '{pipeline_name}' compiled successfully!")
        print(f"Pipeline file: {os.path.abspath(pipeline_file)}")
        print(f"Input table: {input_table}")
        print(f"Hive Host: {hive_host}")
        
        return pipeline_file
        
    except Exception as e:
        print(f"Error compiling pipeline: {str(e)}")
        raise

def main():
    """CLI for compilation."""
    parser = argparse.ArgumentParser(description="Siamese Pipeline")
    
    # Action flags
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline")
    
    # Pipeline parameters (optional with defaults)
    parser.add_argument("--input-table", default="preprocessed_analytics.model_reference", 
                       help="Input Hive table")
    parser.add_argument("--hive-host", default="172.17.235.21", help="Hive server host")
    parser.add_argument("--output", default="lstm-siamese-pipeline.yaml", help="Output pipeline file")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    global CACHE_ENABLED
    if args.no_cache:
        CACHE_ENABLED = False
    
    if args.compile:
        pipeline_file = compile_pipeline(
            input_table=args.input_table,
            hive_host=args.hive_host,
            pipeline_file=args.output
        )
        print(f"\nSteps:")
        print("1. Extract Data")
        print("2. Run Siamese")
        print("3. Save Results")
        print(f"\nUsage: Upload {args.output} to your Kubeflow Pipelines UI")
        return pipeline_file
    else:
        print("Use --compile flag to compile the pipeline")
        print("Example: python lstm_siamese_kubeflow_pipeline.py --compile")
        return None

if __name__ == "__main__":
    main()