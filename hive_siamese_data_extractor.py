import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pyhive import hive
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


class HiveSiameseDataExtractor:
    """
    Class to extract data from Hive tables and prepare for LSTM Siamese text similarity pipeline.
    """
    
    def __init__(self, host: str, port: int = 10000, username: str = None, database: str = "default"):
        """Initialize Hive connection parameters."""
        self.host = host
        self.port = port
        self.username = username
        self.database = database
        self.connection = None
    
    def connect(self):
        """Establish connection to Hive."""
        try:
            self.connection = hive.Connection(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.username,
                auth='NOSASL'
            )
            print(f"✅ Connected to Hive: {self.host}:{self.port}/{self.database}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Hive: {e}")
            return False
    
    def disconnect(self):
        """Close Hive connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Hive connection closed")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        if not self.connection:
            raise RuntimeError("Not connected to Hive")
        
        try:
            df = pd.read_sql(query, self.connection)
            print(f"📊 Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    def detect_table_structure(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect if table has _left/_right columns (production) or single columns (testing).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with structure information
        """
        columns = list(df.columns)
        
        # Remove table prefixes first to analyze column structure
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
        
        if matching_fields:
            structure_type = "production"
            message = f"🏭 Production table detected with {len(matching_fields)} matching field pairs"
        else:
            structure_type = "testing"
            message = f"🧪 Testing table detected with {len(clean_columns)} fields for self-matching"
        
        return {
            'type': structure_type,
            'columns': columns,
            'clean_columns': clean_columns,
            'left_columns': left_columns,
            'right_columns': right_columns,
            'matching_fields': list(matching_fields),
            'message': message
        }
    
    def convert_production_format(self, df: pd.DataFrame, structure: Dict) -> pd.DataFrame:
        """Convert production table with _left/_right columns to LSTM Siamese format."""
        records = []
        
        for idx, row in df.iterrows():
            # Build left and right sentences
            left_parts = []
            right_parts = []
            
            for field in structure['matching_fields']:
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
            
            # For production, determine similarity
            # This could be based on exact match, fuzzy match, or existing label column
            is_similar = 1 if left_text.lower().strip() == right_text.lower().strip() else 0
            
            # Check if there's an existing similarity/match column
            similarity_cols = [col for col in df.columns if any(sim_term in col.lower() 
                             for sim_term in ['similar', 'match', 'label', 'ground_truth'])]
            if similarity_cols:
                # Use existing ground truth if available
                is_similar = int(row[similarity_cols[0]]) if pd.notna(row[similarity_cols[0]]) else is_similar
            
            record = {
                'sentences1': left_text,
                'sentences2': right_text,
                'is_similar': is_similar
            }
            records.append(record)
        
        result_df = pd.DataFrame(records)
        print(f"✅ Successfully converted {len(records)} production records with left/right pairs")
        return result_df
    
    def convert_testing_format(self, df: pd.DataFrame, structure: Dict) -> pd.DataFrame:
        """Convert testing table for self-matching to LSTM Siamese format."""
        records = []
        
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
            
            # Create record for self-matching (similar to itself)
            record = {
                'sentences1': record_text,
                'sentences2': record_text,  # Same record for self-matching
                'is_similar': 1  # Same record is always similar
            }
            records.append(record)
        
        result_df = pd.DataFrame(records)
        print(f"✅ Successfully converted {len(records)} testing records for self-matching")
        return result_df
    
    def create_text_pairs(self, df: pd.DataFrame, mode: str = 'auto') -> pd.DataFrame:
        """
        Create text pairs for LSTM Siamese training from different sources.
        
        Args:
            df: Input DataFrame
            mode: 'auto', 'production', 'testing', or 'cross_product'
            
        Returns:
            DataFrame with sentences1, sentences2, is_similar columns
        """
        if mode == 'cross_product':
            return self.create_cross_product_pairs(df)
        
        # Detect table structure
        structure = self.detect_table_structure(df)
        print(structure['message'])
        
        if structure['type'] == 'production' and mode != 'testing':
            return self.convert_production_format(df, structure)
        else:
            return self.convert_testing_format(df, structure)
    
    def create_cross_product_pairs(self, df: pd.DataFrame, max_pairs: int = 10000) -> pd.DataFrame:
        """
        Create all possible pairs from records (cross product) for training data generation.
        
        Args:
            df: Input DataFrame
            max_pairs: Maximum number of pairs to generate
            
        Returns:
            DataFrame with sentence pairs and similarity labels
        """
        records = []
        
        # Convert each row to text
        text_records = []
        for idx, row in df.iterrows():
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    clean_col = col.split('.', 1)[1] if '.' in col else col
                    value = str(row[col]).strip()
                    text_parts.append(f"{clean_col}: {value}")
            text_records.append(" ".join(text_parts))
        
        print(f"🔄 Generating pairs from {len(text_records)} records...")
        
        # Generate pairs
        pair_count = 0
        for i in range(len(text_records)):
            for j in range(i, len(text_records)):
                if pair_count >= max_pairs:
                    break
                
                # Determine similarity (same record = similar, different = not similar)
                is_similar = 1 if i == j else 0
                
                record = {
                    'sentences1': text_records[i],
                    'sentences2': text_records[j],
                    'is_similar': is_similar
                }
                records.append(record)
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        result_df = pd.DataFrame(records)
        print(f"✅ Generated {len(records)} pairs (similar: {result_df['is_similar'].sum()}, "
              f"dissimilar: {len(records) - result_df['is_similar'].sum()})")
        
        return result_df
    
    def extract_and_convert(self, 
                          table_name: str, 
                          output_path: str,
                          sample_limit: Optional[int] = None,
                          matching_mode: str = 'auto',
                          balance_classes: bool = True) -> str:
        """
        Extract data from Hive table and convert to LSTM Siamese format.
        
        Args:
            table_name: Hive table name
            output_path: Path to save the converted data
            sample_limit: Limit number of records (None for all)
            matching_mode: 'auto', 'production', 'testing', or 'cross_product'
            balance_classes: Whether to balance similar/dissimilar pairs
            
        Returns:
            Path to the saved file
        """
        print(f"🔄 Extracting data from table: {table_name}")
        
        # Build query
        if sample_limit:
            query = f"SELECT * FROM {table_name} LIMIT {sample_limit}"
        else:
            query = f"SELECT * FROM {table_name}"
        
        # Execute query
        df = self.execute_query(query)
        print(f"📊 Extracted {len(df)} records")
        
        # Convert to Siamese format
        print(f"🔄 Converting to LSTM Siamese format (mode: {matching_mode})")
        siamese_df = self.create_text_pairs(df, mode=matching_mode)
        
        # Balance classes if requested
        if balance_classes and matching_mode in ['cross_product']:
            siamese_df = self.balance_dataset(siamese_df)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        siamese_df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(siamese_df)} pairs to: {output_path}")
        
        # Show statistics
        similar_count = siamese_df['is_similar'].sum()
        total_count = len(siamese_df)
        print(f"📊 Dataset statistics:")
        print(f"  Total pairs: {total_count}")
        print(f"  Similar pairs: {similar_count} ({similar_count/total_count:.1%})")
        print(f"  Dissimilar pairs: {total_count - similar_count} ({(total_count - similar_count)/total_count:.1%})")
        
        return output_path
    
    def balance_dataset(self, df: pd.DataFrame, ratio: float = 0.5) -> pd.DataFrame:
        """
        Balance the dataset to have a specified ratio of similar/dissimilar pairs.
        
        Args:
            df: Input DataFrame with is_similar column
            ratio: Target ratio of similar pairs (0.5 = balanced)
            
        Returns:
            Balanced DataFrame
        """
        similar_df = df[df['is_similar'] == 1]
        dissimilar_df = df[df['is_similar'] == 0]
        
        similar_count = len(similar_df)
        dissimilar_count = len(dissimilar_df)
        
        print(f"🎯 Balancing dataset (current: {similar_count} similar, {dissimilar_count} dissimilar)")
        
        if ratio == 0.5:
            # Equal balance
            target_count = min(similar_count, dissimilar_count)
            balanced_similar = similar_df.sample(n=target_count, random_state=42)
            balanced_dissimilar = dissimilar_df.sample(n=target_count, random_state=42)
        else:
            # Custom ratio
            total_target = min(len(df), max(similar_count, dissimilar_count) * 2)
            similar_target = int(total_target * ratio)
            dissimilar_target = total_target - similar_target
            
            balanced_similar = similar_df.sample(n=min(similar_target, similar_count), 
                                               random_state=42, replace=similar_target > similar_count)
            balanced_dissimilar = dissimilar_df.sample(n=min(dissimilar_target, dissimilar_count), 
                                                     random_state=42, replace=dissimilar_target > dissimilar_count)
        
        balanced_df = pd.concat([balanced_similar, balanced_dissimilar], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"✅ Balanced dataset: {len(balanced_df)} pairs "
              f"({balanced_df['is_similar'].sum()} similar, {len(balanced_df) - balanced_df['is_similar'].sum()} dissimilar)")
        
        return balanced_df


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Extract data from Hive for LSTM Siamese text similarity')
    
    parser.add_argument('--host', required=True, help='Hive host')
    parser.add_argument('--port', type=int, default=10000, help='Hive port')
    parser.add_argument('--username', help='Hive username')
    parser.add_argument('--database', default='default', help='Hive database')
    parser.add_argument('--table', required=True, help='Input table name')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--limit', type=int, help='Limit number of records')
    parser.add_argument('--mode', default='auto', 
                       choices=['auto', 'production', 'testing', 'cross_product'],
                       help='Matching mode')
    parser.add_argument('--balance', action='store_true', help='Balance similar/dissimilar pairs')
    parser.add_argument('--config', help='JSON config file path')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Override command line args with config
        for key, value in config.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
    
    # Create extractor
    extractor = HiveSiameseDataExtractor(
        host=args.host,
        port=args.port,
        username=args.username,
        database=args.database
    )
    
    try:
        # Connect to Hive
        if not extractor.connect():
            exit(1)
        
        # Extract and convert data
        output_path = extractor.extract_and_convert(
            table_name=args.table,
            output_path=args.output,
            sample_limit=args.limit,
            matching_mode=args.mode,
            balance_classes=args.balance
        )
        
        print(f"🎉 Successfully created LSTM Siamese dataset: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    finally:
        extractor.disconnect()


if __name__ == "__main__":
    main()