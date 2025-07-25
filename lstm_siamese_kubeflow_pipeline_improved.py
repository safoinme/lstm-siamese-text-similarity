from datetime import datetime
from kfp import compiler, dsl
from typing import NamedTuple, Optional
import os
import argparse
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar

CACHE_ENABLED = True

def safe_table_name(table: str) -> str:
    """Return the table part of a fully-qualified hive table name."""
    return table.split('.')[-1]

def extract_and_process_siamese_func(
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    input_table: str,
    sample_limit: Optional[int] = None,
    matching_mode: str = 'auto',
    # Siamese parameters
    model_path: str = "/models/siamese_model.h5",
    max_sequence_length: int = 100,
    threshold: float = 0.5,
    use_gpu: bool = False
) -> dict:
    """Extract from Hive, process with LSTM Siamese for text similarity, all in one step."""
    from pyhive import hive
    import pandas as pd
    import json
    import os
    import numpy as np
    import tempfile
    
    try:
        # GPU Detection and Setup
        if use_gpu:
            print("=== GPU SETUP ===")
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    print(f"GPU Available: {len(gpus)} GPU(s) found")
                    # Configure GPU memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    print("WARNING: GPU requested but not available, falling back to CPU")
                    use_gpu = False
            except Exception as e:
                print(f"WARNING: TensorFlow GPU setup failed: {str(e)}, falling back to CPU")
                use_gpu = False
        else:
            print("=== CPU MODE ===")
        
        print("=== STEP 1: Extract from Hive ===")
        # Connect to Hive
        connection = hive.Connection(
            host=hive_host,
            port=hive_port,
            username=hive_user,
            database=hive_database
        )
        
        # Extract data
        query = f"SELECT * FROM {input_table}"
        if sample_limit:
            query += f" LIMIT {sample_limit}"
        
        df = pd.read_sql(query, connection)
        print(f"Extracted {len(df)} records from {input_table}")
        connection.close()
        
        print("=== STEP 2: Convert to Siamese format ===")
        # Convert to Siamese format (reusing existing logic)
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
                    'sentences1': left_text,
                    'sentences2': right_text,
                    'record_id': idx
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
                        col_val_parts.append(f"{clean_col}: {value}")
                
                record_text = " ".join(col_val_parts)
                record = {
                    'sentences1': record_text,
                    'sentences2': record_text,  # Self-comparison for testing
                    'record_id': idx
                }
                records.append(record)
            
            return records
        
        def convert_to_siamese_format(df, matching_mode):
            structure = detect_table_structure(df)
            
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
        print(f"Converted {len(siamese_records)} records to Siamese format")
        
        print("=== STEP 3: Run LSTM Siamese Matching ===")
        # Check if model files exist
        model_files = {
            'model': model_path,
            'tokenizer': model_path.replace('.h5', '_tokenizer.pkl'),
            'config': model_path.replace('.h5', '_config.json')
        }
        
        for file_type, file_path in model_files.items():
            if not os.path.exists(file_path):
                print(f"Warning: {file_type} file not found at {file_path}")
        
        # Load model and dependencies
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            import pickle
            
            print(f"Loading model from {model_path}")
            model = load_model(model_path)
            
            # Load tokenizer
            tokenizer_path = model_path.replace('.h5', '_tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Prepare data for prediction
            sentences1 = [record['sentences1'] for record in siamese_records]
            sentences2 = [record['sentences2'] for record in siamese_records]
            
            print(f"Making predictions for {len(siamese_records)} pairs...")
            
            # Convert texts to sequences
            seq1 = tokenizer.texts_to_sequences(sentences1)
            seq2 = tokenizer.texts_to_sequences(sentences2)
            
            # Pad sequences
            seq1 = pad_sequences(seq1, maxlen=max_sequence_length)
            seq2 = pad_sequences(seq2, maxlen=max_sequence_length)
            
            # Create leaks features (simple features like length difference)
            leaks = []
            for s1, s2 in zip(sentences1, sentences2):
                leak_features = [
                    len(s1), len(s2),
                    abs(len(s1) - len(s2)),
                    len(set(s1.split()).intersection(set(s2.split())))
                ]
                leaks.append(leak_features)
            
            leaks = np.array(leaks)
            
            # Make predictions
            predictions = model.predict([seq1, seq2, leaks])
            predictions_binary = (predictions.flatten() > threshold).astype(int)
            
            # Process results
            results = []
            metrics = {"total_pairs": 0, "matches": 0, "non_matches": 0}
            
            for i, record in enumerate(siamese_records):
                result_data = {
                    'record_id': record['record_id'],
                    'sentences1': record['sentences1'],
                    'sentences2': record['sentences2'],
                    'similarity_score': float(predictions[i][0]),
                    'match': int(predictions_binary[i]),
                    'match_confidence': float(predictions[i][0]),
                    'model_type': 'lstm_siamese',
                    'threshold_used': threshold
                }
                results.append(result_data)
                
                metrics["total_pairs"] += 1
                if result_data['match'] == 1:
                    metrics["matches"] += 1
                else:
                    metrics["non_matches"] += 1
            
            print(f"Processing completed. Metrics: {metrics}")
            print(f"Match rate: {np.mean(predictions_binary):.2%}")
            print(f"Average similarity score: {np.mean(predictions.flatten()):.3f}")
            
            return {
                "results": results,
                "metrics": metrics,
                "gpu_used": use_gpu,
                "status": "success"
            }
            
        except Exception as e:
            print(f"ERROR in LSTM Siamese processing: {str(e)}")
            # Return basic structure matching results if model fails
            basic_results = []
            for i, record in enumerate(siamese_records):
                # Simple fallback: exact match
                is_exact_match = record['sentences1'].lower().strip() == record['sentences2'].lower().strip()
                result_data = {
                    'record_id': record['record_id'],
                    'sentences1': record['sentences1'],
                    'sentences2': record['sentences2'],
                    'similarity_score': 1.0 if is_exact_match else 0.0,
                    'match': 1 if is_exact_match else 0,
                    'match_confidence': 1.0 if is_exact_match else 0.0,
                    'model_type': 'fallback_exact_match',
                    'threshold_used': threshold
                }
                basic_results.append(result_data)
            
            return {
                "results": basic_results,
                "metrics": {"total_pairs": len(basic_results), "matches": sum(r['match'] for r in basic_results), "non_matches": len(basic_results) - sum(r['match'] for r in basic_results)},
                "gpu_used": False,
                "status": f"fallback_mode: {str(e)}"
            }
                    
    except Exception as e:
        print(f"ERROR in extract_and_process_siamese_func: {str(e)}")
        return {
            "results": [],
            "metrics": {"total_pairs": 0, "matches": 0, "non_matches": 0},
            "gpu_used": False,
            "status": f"error: {str(e)}"
        }

def save_results_to_hive_func(
    processing_results: dict,
    hive_host: str,
    hive_port: int,
    hive_user: str,
    hive_database: str,
    output_table: str,
    save_results: bool = True
) -> str:
    """Save processing results to Hive."""
    if not save_results:
        print("Skipping Hive save as save_results is False")
        return "Skipped"
    
    if processing_results.get("status") != "success" and not processing_results.get("status", "").startswith("fallback_mode"):
        return f"ERROR: Cannot save results due to processing error: {processing_results.get('status', 'unknown error')}"
    
    from pyhive import hive
    import pandas as pd
    import json
    import tempfile
    import os
    
    try:
        results_data = processing_results.get("results", [])
        if not results_data:
            print("No results to save")
            return "No results"
        
        # Convert results to DataFrame
        processed_results = []
        for result in results_data:
            processed_results.append({
                'record_id': result.get('record_id', 0),
                'sentences1': str(result.get('sentences1', ''))[:1000],  # Truncate long text
                'sentences2': str(result.get('sentences2', ''))[:1000],
                'similarity_score': result.get('similarity_score', 0.0),
                'match': result.get('match', 0),
                'match_confidence': result.get('match_confidence', 0.0),
                'model_type': result.get('model_type', 'lstm_siamese'),
                'threshold_used': result.get('threshold_used', 0.5),
                'processing_timestamp': datetime.now().isoformat(),
                'gpu_used': processing_results.get('gpu_used', False)
            })
        
        df = pd.DataFrame(processed_results)
        
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
            sentences1 STRING,
            sentences2 STRING,
            similarity_score DOUBLE,
            match INT,
            match_confidence DOUBLE,
            model_type STRING,
            threshold_used DOUBLE,
            processing_timestamp STRING,
            gpu_used BOOLEAN
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
            
            gpu_status = "with GPU" if processing_results.get('gpu_used') else "with CPU"
            model_status = processing_results.get('status', 'success')
            print(f"Successfully saved {len(processed_results)} results to {output_table} (processed {gpu_status}, status: {model_status})")
            return f"Saved {len(processed_results)} results to {output_table} (processed {gpu_status})"
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        connection.close()
        
    except Exception as e:
        print(f"ERROR saving to Hive: {str(e)}")
        return f"Error: {str(e)}"

# Create Kubeflow components
extract_and_process_siamese_op = create_component_from_func(
    func=extract_and_process_siamese_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',  # Custom image with TensorFlow + Hive
)

save_results_to_hive_op = create_component_from_func(
    func=save_results_to_hive_func,
    base_image='172.17.232.16:9001/lstm-siamese:2.0',
)

@dsl.pipeline(
    name="lstm-siamese-text-similarity",
    description="LSTM Siamese Text Similarity Pipeline with Kubeflow and Hive integration"
)
def lstm_siamese_text_similarity_pipeline(
    # Hive connection parameters
    hive_host: str = "172.17.235.21",
    hive_port: int = 10000,
    hive_user: str = "lhimer",
    hive_database: str = "preprocessed_analytics",
    
    # Input table
    input_table: str = "preprocessed_analytics.model_reference",
    
    # Data limits (for testing)
    sample_limit: Optional[int] = None,
    
    # Matching mode
    matching_mode: str = 'auto',
    
    # Siamese model parameters
    model_path: str = "/home/jovyan/models/siamese_model.h5",
    max_sequence_length: int = 100,
    threshold: float = 0.5,
    use_gpu: bool = False,
    
    # Output parameters
    save_to_hive: bool = False,
    output_table: str = "lstm_siamese_results"
):
    """
    LSTM Siamese Text Similarity pipeline for production use
    """
    
    # Define environment variables for Hive connectivity
    env_vars = [
        V1EnvVar(name='HIVE_HOST', value=hive_host),
        V1EnvVar(name='HIVE_PORT', value=str(hive_port)),
        V1EnvVar(name='HIVE_USER', value=hive_user),
        V1EnvVar(name='HIVE_DATABASE', value=hive_database)
    ]
    
    # Step 1: Extract from Hive and process with LSTM Siamese
    process_results = extract_and_process_siamese_op(
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        input_table=input_table,
        sample_limit=sample_limit,
        matching_mode=matching_mode,
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        threshold=threshold,
        use_gpu=use_gpu
    )
    
    # Add environment variables
    for env_var in env_vars:
        process_results.add_env_variable(env_var)
    
    process_results.set_display_name('Extract from Hive and Run LSTM Siamese')
    
    # Force image pull
    process_results.container.set_image_pull_policy('Always')
    
    # GPU configuration
    if use_gpu:
        process_results.set_gpu_limit(1)
        process_results.set_memory_limit('16Gi')
        process_results.set_memory_request('8Gi')
    else:
        process_results.set_memory_limit('8Gi')
        process_results.set_memory_request('4Gi')
    
    process_results.set_cpu_limit('4')
    process_results.set_cpu_request('2')
    process_results.set_caching_options(enable_caching=False)  # Don't cache for production
    
    # Step 2: Save results to Hive
    save_results = save_results_to_hive_op(
        processing_results=process_results.output,
        hive_host=hive_host,
        hive_port=hive_port,
        hive_user=hive_user,
        hive_database=hive_database,
        output_table=output_table,
        save_results=save_to_hive
    ).after(process_results)
    
    # Add environment variables (CPU-only step)
    for env_var in env_vars:
        save_results.add_env_variable(env_var)
    save_results.set_display_name('Save Results to Hive')
    save_results.set_caching_options(enable_caching=False)
    
    # Force image pull for save step too
    save_results.container.set_image_pull_policy('Always')

def compile_pipeline(
    input_table: str = "preprocessed_analytics.model_reference",
    hive_host: str = "172.17.235.21",
    pipeline_file: str = "lstm-siamese-pipeline.yaml"
):
    """Compile the LSTM Siamese text similarity pipeline."""
    try:
        compiler.Compiler().compile(
            pipeline_func=lstm_siamese_text_similarity_pipeline,
            package_path=pipeline_file,
            type_check=True
        )
        
        print(f"\nLSTM Siamese Pipeline compiled successfully!")
        print(f"Pipeline file: {os.path.abspath(pipeline_file)}")
        print(f"Input table: {input_table}")
        print(f"Hive Host: {hive_host}")
        print("Features:")
        print("   - LSTM Siamese Neural Network for text similarity")
        print("   - Hive integration for data input/output")
        print("   - GPU support with automatic fallback")
        print("   - Production and testing modes")
        
        return pipeline_file
        
    except Exception as e:
        print(f"ERROR compiling pipeline: {str(e)}")
        raise

def main():
    """Command line interface for LSTM Siamese pipeline compilation."""
    parser = argparse.ArgumentParser(description="LSTM Siamese Text Similarity Kubeflow Pipeline")
    
    parser.add_argument("--compile", action="store_true", help="Compile the pipeline")
    parser.add_argument("--input-table", default="preprocessed_analytics.model_reference", help="Input Hive table")
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
        print(f"\nPipeline Steps:")
        print("1. Extract from Hive + Run LSTM Siamese - Extract data and run text similarity")
        print("2. Save Results to Hive - Store similarity scores and predictions")
        print(f"\nUsage: Upload {args.output} to your Kubeflow Pipelines UI")
        print("Training: Use the Jupyter notebook for model training")
        return pipeline_file
    else:
        print("Use --compile flag to compile the pipeline")
        return None

if __name__ == "__main__":
    main()