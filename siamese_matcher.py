#!/usr/bin/env python3
"""
LSTM Siamese Text Similarity Matcher
Similar to DITTO's matcher.py but for LSTM Siamese neural networks.
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import json
from datetime import datetime

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


class SiameseMatcher:
    """LSTM Siamese Neural Network for Text Similarity Matching."""
    
    def __init__(self, config: dict):
        """
        Initialize the Siamese matcher.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.is_trained = False
    
    def prepare_data(self, sentences1: List[str], sentences2: List[str], labels: Optional[List[int]] = None) -> Tuple:
        """
        Prepare text data for training or prediction.
        
        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            labels: List of similarity labels (optional)
            
        Returns:
            Tuple of processed sequences and labels
        """
        print(f"📊 Preparing {len(sentences1)} sentence pairs...")
        
        # Create or use existing tokenizer
        if self.tokenizer is None:
            print("🔤 Creating tokenizer...")
            all_sentences = sentences1 + sentences2
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(all_sentences)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            print(f"📖 Vocabulary size: {self.vocab_size}")
        
        # Convert texts to sequences
        seq1 = self.tokenizer.texts_to_sequences(sentences1)
        seq2 = self.tokenizer.texts_to_sequences(sentences2)
        
        # Pad sequences
        max_len = self.config.get('max_sequence_length', 100)
        seq1 = pad_sequences(seq1, maxlen=max_len)
        seq2 = pad_sequences(seq2, maxlen=max_len)
        
        print(f"🔢 Sequences padded to length {max_len}")
        
        if labels is not None:
            return seq1, seq2, np.array(labels)
        else:
            return seq1, seq2
    
    def build_model(self) -> Model:
        """
        Build the LSTM Siamese neural network model.
        
        Returns:
            Compiled Keras model
        """
        print("🏗️  Building LSTM Siamese model...")
        
        # Model parameters
        embedding_dim = self.config.get('embedding_dim', 300)
        max_len = self.config.get('max_sequence_length', 100)
        lstm_units = self.config.get('number_lstm', 50)
        dropout_lstm = self.config.get('rate_drop_lstm', 0.25)
        dense_units = self.config.get('number_dense_units', 50)
        dropout_dense = self.config.get('rate_drop_dense', 0.25)
        activation = self.config.get('activation_function', 'relu')
        
        # Input layers
        input1 = Input(shape=(max_len,), name='input1')
        input2 = Input(shape=(max_len,), name='input2')
        
        # Shared embedding layer
        embedding = Embedding(self.vocab_size, embedding_dim, name='embedding')
        
        # Shared LSTM layer
        lstm = Bidirectional(LSTM(lstm_units, dropout=dropout_lstm), name='lstm')
        
        # Process both inputs through shared layers
        embed1 = embedding(input1)
        embed2 = embedding(input2)
        
        lstm1 = lstm(embed1)
        lstm2 = lstm(embed2)
        
        # Calculate absolute difference
        distance = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name='distance'
        )([lstm1, lstm2])
        
        # Dense layers for classification
        dense = Dense(dense_units, activation=activation, name='dense')(distance)
        dense = Dropout(dropout_dense, name='dropout')(dense)
        output = Dense(1, activation='sigmoid', name='output')(dense)
        
        # Create and compile model
        model = Model(inputs=[input1, input2], outputs=output, name='siamese_lstm')
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        print("✅ Model built successfully")
        print(f"📊 Model summary:")
        model.summary()
        
        return model
    
    def train(self, sentences1: List[str], sentences2: List[str], labels: List[int]) -> dict:
        """
        Train the LSTM Siamese model.
        
        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            labels: List of similarity labels
            
        Returns:
            Training history
        """
        print("🚀 Starting training...")
        
        # Prepare data
        seq1, seq2, labels_array = self.prepare_data(sentences1, sentences2, labels)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Training parameters
        validation_split = self.config.get('validation_split', 0.2)
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 64)
        
        # Split data
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            seq1, seq2, labels_array, 
            test_size=validation_split, 
            random_state=42,
            stratify=labels_array
        )
        
        print(f"📊 Training set: {len(X1_train)} pairs")
        print(f"📊 Validation set: {len(X1_val)} pairs")
        
        # Train model
        history = self.model.fit(
            [X1_train, X2_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([X1_val, X2_val], y_val),
            verbose=1
        )
        
        self.is_trained = True
        print("✅ Training completed!")
        
        # Print final metrics
        final_metrics = {
            'train_loss': history.history['loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
        
        print(f"📊 Final metrics: {final_metrics}")
        
        return history.history
    
    def predict(self, sentences1: List[str], sentences2: List[str]) -> np.ndarray:
        """
        Predict similarity scores for sentence pairs.
        
        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            
        Returns:
            Array of similarity scores
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        print(f"🔮 Making predictions for {len(sentences1)} pairs...")
        
        # Prepare data
        seq1, seq2 = self.prepare_data(sentences1, sentences2)
        
        # Make predictions
        predictions = self.model.predict([seq1, seq2], verbose=1)
        
        print("✅ Predictions completed!")
        
        return predictions.flatten()
    
    def save_model(self, model_path: str):
        """
        Save the trained model and tokenizer.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save tokenizer
        tokenizer_path = model_path.replace('.h5', '_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"💾 Tokenizer saved to: {tokenizer_path}")
        
        # Save config
        config_path = model_path.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"💾 Config saved to: {config_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model and tokenizer.
        
        Args:
            model_path: Path to the saved model
        """
        # Load model
        self.model = load_model(model_path)
        print(f"📥 Model loaded from: {model_path}")
        
        # Load tokenizer
        tokenizer_path = model_path.replace('.h5', '_tokenizer.pkl')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"📥 Tokenizer loaded from: {tokenizer_path}")
        else:
            print("⚠️  Tokenizer not found - you'll need to retrain or provide tokenizer")
        
        # Load config if available
        config_path = model_path.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
            print(f"📥 Config loaded from: {config_path}")
        
        self.is_trained = True


def load_data(input_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Load data from CSV file.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        Tuple of sentences1, sentences2, and labels
    """
    print(f"📥 Loading data from: {input_path}")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        sentences1 = df['sentences1'].tolist()
        sentences2 = df['sentences2'].tolist()
        labels = df['is_similar'].tolist() if 'is_similar' in df.columns else None
    else:
        raise ValueError("Only CSV format is currently supported")
    
    print(f"📊 Loaded {len(sentences1)} sentence pairs")
    
    return sentences1, sentences2, labels


def save_results(predictions: np.ndarray, 
                sentences1: List[str], 
                sentences2: List[str], 
                output_path: str,
                threshold: float = 0.5,
                labels: Optional[List[int]] = None):
    """
    Save prediction results to file.
    
    Args:
        predictions: Array of similarity scores
        sentences1: List of first sentences
        sentences2: List of second sentences
        output_path: Path to save results
        threshold: Threshold for binary classification
        labels: Ground truth labels (optional)
    """
    print(f"💾 Saving results to: {output_path}")
    
    # Create results DataFrame
    results = {
        'sentences1': sentences1,
        'sentences2': sentences2,
        'similarity_score': predictions,
        'prediction': (predictions > threshold).astype(int),
        'timestamp': datetime.now().isoformat()
    }
    
    if labels is not None:
        results['ground_truth'] = labels
        results['correct'] = (results['prediction'] == np.array(labels)).astype(int)
    
    results_df = pd.DataFrame(results)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    if output_path.endswith('.csv'):
        results_df.to_csv(output_path, index=False)
    elif output_path.endswith('.json'):
        results_df.to_json(output_path, orient='records', indent=2)
    else:
        # Default to CSV
        results_df.to_csv(output_path, index=False)
    
    print(f"✅ Results saved: {len(results_df)} predictions")
    
    # Print summary statistics
    print(f"📊 Results summary:")
    print(f"  Average similarity score: {np.mean(predictions):.3f}")
    print(f"  Predictions above threshold ({threshold}): {np.sum(predictions > threshold)}")
    print(f"  Match rate: {np.mean(predictions > threshold):.2%}")
    
    if labels is not None:
        accuracy = np.mean(results_df['correct'])
        print(f"  Accuracy: {accuracy:.2%}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='LSTM Siamese Text Similarity Matcher')
    
    # Data arguments
    parser.add_argument('--input_path', required=True, help='Path to input CSV file')
    parser.add_argument('--output_path', required=True, help='Path to save results')
    
    # Model arguments
    parser.add_argument('--model_path', help='Path to save/load model')
    parser.add_argument('--mode', choices=['train', 'predict', 'train_predict'], 
                       default='train_predict', help='Operation mode')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--max_sequence_length', type=int, default=100, help='Max sequence length')
    parser.add_argument('--number_lstm', type=int, default=50, help='LSTM units')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    # Optional parameters
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--rate_drop_lstm', type=float, default=0.25, help='LSTM dropout rate')
    parser.add_argument('--rate_drop_dense', type=float, default=0.25, help='Dense dropout rate')
    parser.add_argument('--config', help='JSON config file')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    config = vars(args)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    print("🚀 Starting LSTM Siamese Text Similarity Matcher")
    print(f"📋 Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Load data
        sentences1, sentences2, labels = load_data(args.input_path)
        
        # Initialize matcher
        matcher = SiameseMatcher(config)
        
        if args.mode in ['train', 'train_predict']:
            if labels is None:
                raise ValueError("Training requires labels in the input data")
            
            print("🏋️  Training mode")
            
            # Train model
            history = matcher.train(sentences1, sentences2, labels)
            
            # Save model if path provided
            if args.model_path:
                matcher.save_model(args.model_path)
        
        elif args.mode == 'predict':
            if not args.model_path or not os.path.exists(args.model_path):
                raise ValueError("Prediction mode requires existing model path")
            
            print("🔮 Prediction mode")
            
            # Load model
            matcher.load_model(args.model_path)
        
        if args.mode in ['predict', 'train_predict']:
            # Make predictions
            predictions = matcher.predict(sentences1, sentences2)
            
            # Save results
            save_results(predictions, sentences1, sentences2, args.output_path, 
                        args.threshold, labels)
        
        print("🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())