#!/usr/bin/env python3
"""
Demo script showing how to use the LSTM Siamese Text Similarity system
Similar to the DITTO demo but for LSTM Siamese neural networks.
"""

import os
import pandas as pd
from datetime import datetime

def create_sample_data():
    """Create sample data for demonstration."""
    print("📊 Creating sample data...")
    
    # Sample text pairs with similarity labels
    sample_data = {
        'sentences1': [
            'John Smith works at Microsoft Corporation',
            'Mary Johnson is a school teacher',
            'The quick brown fox jumps over the lazy dog',
            'Apple Inc. is a technology company',
            'Barcelona is a beautiful city in Spain',
            'Machine learning is a subset of artificial intelligence',
            'The weather is sunny today',
            'Python is a programming language',
            'Coffee shops are popular meeting places',
            'Electric cars are becoming more common'
        ],
        'sentences2': [
            'Jon Smith employed by Microsoft Corp',
            'Maria Johnson teaches at elementary school',
            'A fast brown fox leaps over a sleeping dog',
            'Apple Incorporated is a tech company',
            'Barcelona is a lovely Spanish city',
            'ML is part of AI technology',
            'Today has bright sunshine',
            'Python programming language is popular',
            'Coffee houses are great for meetings',
            'Electric vehicles are increasingly popular'
        ],
        'is_similar': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # All pairs are similar
    }
    
    # Add some dissimilar pairs for balance
    dissimilar_data = {
        'sentences1': [
            'Dogs are loyal pets',
            'Summer vacation is relaxing',
            'Mathematics is challenging',
            'Pizza is delicious food',
            'Mountains are tall'
        ],
        'sentences2': [
            'Cars need regular maintenance',
            'Winter sports are exciting',
            'History is fascinating',
            'Computers are powerful tools',
            'Oceans are deep'
        ],
        'is_similar': [0, 0, 0, 0, 0]  # All pairs are dissimilar
    }
    
    # Combine datasets
    all_data = {
        'sentences1': sample_data['sentences1'] + dissimilar_data['sentences1'],
        'sentences2': sample_data['sentences2'] + dissimilar_data['sentences2'],
        'is_similar': sample_data['is_similar'] + dissimilar_data['is_similar']
    }
    
    df = pd.DataFrame(all_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Created {len(df)} text pairs")
    print(f"   Similar pairs: {df['is_similar'].sum()}")
    print(f"   Dissimilar pairs: {len(df) - df['is_similar'].sum()}")
    
    return df

def demo_data_extraction():
    """Demonstrate data extraction capabilities."""
    print("\n" + "="*60)
    print("🔄 DEMO: Data Extraction")
    print("="*60)
    
    try:
        from hive_siamese_data_extractor import HiveSiameseDataExtractor
        
        print("✅ HiveSiameseDataExtractor imported successfully")
        print("💡 In production, you would connect to your Hive cluster:")
        print("   extractor = HiveSiameseDataExtractor(host='your-hive-host', port=10000)")
        print("   extractor.connect()")
        print("   data = extractor.extract_and_convert('your_table', 'output.csv')")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

def demo_model_training():
    """Demonstrate model training and prediction."""
    print("\n" + "="*60)
    print("🔄 DEMO: Model Training and Prediction")
    print("="*60)
    
    try:
        from siamese_matcher import SiameseMatcher
        
        # Create sample data
        df = create_sample_data()
        
        # Save sample data
        sample_file = 'demo_data.csv'
        df.to_csv(sample_file, index=False)
        print(f"💾 Sample data saved to: {sample_file}")
        
        # Configuration for quick demo
        config = {
            'embedding_dim': 100,  # Smaller for faster demo
            'max_sequence_length': 50,
            'number_lstm': 25,
            'epochs': 3,  # Few epochs for quick demo
            'batch_size': 8,
            'validation_split': 0.2
        }
        
        print(f"🔧 Using config: {config}")
        
        # Initialize matcher
        matcher = SiameseMatcher(config)
        
        # Prepare data
        sentences1 = df['sentences1'].tolist()
        sentences2 = df['sentences2'].tolist()
        labels = df['is_similar'].tolist()
        
        print(f"📊 Training on {len(sentences1)} sentence pairs...")
        
        # Train model
        print("🚀 Starting training...")
        history = matcher.train(sentences1, sentences2, labels)
        
        print("✅ Training completed!")
        
        # Save model
        model_path = 'demo_siamese_model.h5'
        matcher.save_model(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = matcher.predict(sentences1, sentences2)
        
        # Create results
        results_df = df.copy()
        results_df['similarity_score'] = predictions
        results_df['prediction'] = (predictions > 0.5).astype(int)
        results_df['correct'] = (results_df['prediction'] == results_df['is_similar']).astype(int)
        
        # Save results
        results_file = 'demo_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"💾 Results saved to: {results_file}")
        
        # Show statistics
        accuracy = results_df['correct'].mean()
        avg_similarity = predictions.mean()
        
        print(f"\n📊 Results Summary:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Average similarity score: {avg_similarity:.3f}")
        print(f"   Predictions above 0.5: {sum(predictions > 0.5)}/{len(predictions)}")
        
        print(f"\n📋 Sample Results:")
        sample_results = results_df[['sentences1', 'sentences2', 'similarity_score', 'prediction', 'is_similar']].head()
        for _, row in sample_results.iterrows():
            status = "✅" if row['prediction'] == row['is_similar'] else "❌"
            print(f"   {status} Score: {row['similarity_score']:.3f} | Pred: {row['prediction']} | Actual: {row['is_similar']}")
            print(f"      Text1: {row['sentences1'][:50]}...")
            print(f"      Text2: {row['sentences2'][:50]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training demo: {e}")
        print("💡 This might be due to missing dependencies or insufficient data")
        return False

def demo_kubeflow_pipeline():
    """Demonstrate Kubeflow pipeline compilation."""
    print("\n" + "="*60)
    print("🔄 DEMO: Kubeflow Pipeline")
    print("="*60)
    
    try:
        from lstm_siamese_kubeflow_pipeline import compile_pipeline
        
        print("🔧 Compiling Kubeflow pipeline...")
        pipeline_file = compile_pipeline()
        
        if os.path.exists(pipeline_file):
            print(f"✅ Pipeline compiled successfully: {pipeline_file}")
            print(f"📊 File size: {os.path.getsize(pipeline_file)} bytes")
            print("💡 Upload this YAML file to your Kubeflow Pipelines UI")
        else:
            print("❌ Pipeline file not found after compilation")
        
    except Exception as e:
        print(f"❌ Error compiling pipeline: {e}")
        print("💡 Make sure kfp is installed: pip install kfp==2.0.1")

def demo_command_line_usage():
    """Show command line usage examples."""
    print("\n" + "="*60)
    print("🔄 DEMO: Command Line Usage")
    print("="*60)
    
    print("📋 Example commands for production use:")
    print()
    
    print("1️⃣  Extract data from Hive:")
    print("   python hive_siamese_data_extractor.py \\")
    print("     --host YOUR_HIVE_HOST \\")
    print("     --port 10000 \\")
    print("     --username YOUR_USERNAME \\")
    print("     --database YOUR_DATABASE \\")
    print("     --table YOUR_TABLE \\")
    print("     --output extracted_data.csv \\")
    print("     --limit 1000 \\")
    print("     --mode auto")
    print()
    
    print("2️⃣  Train and predict with LSTM Siamese:")
    print("   python siamese_matcher.py \\")
    print("     --input_path extracted_data.csv \\")
    print("     --output_path predictions.csv \\")
    print("     --model_path siamese_model.h5 \\")
    print("     --mode train_predict \\")
    print("     --epochs 20 \\")
    print("     --batch_size 64")
    print()
    
    print("3️⃣  Compile Kubeflow pipeline:")
    print("   python lstm_siamese_kubeflow_pipeline.py --compile")
    print()
    
    print("4️⃣  Build Docker image:")
    print("   docker build -t your-registry/lstm-siamese-hive:latest .")
    print("   docker push your-registry/lstm-siamese-hive:latest")

def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        'demo_data.csv',
        'demo_results.csv', 
        'demo_siamese_model.h5',
        'demo_siamese_model_tokenizer.pkl',
        'demo_siamese_model_config.json',
        'demo_siamese_model_history.json',
        'lstm_siamese_pipeline.yaml'
    ]
    
    print("\n🧹 Cleaning up demo files...")
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️  Removed: {file}")
    
    print("✅ Cleanup completed!")

def main():
    """Run the complete demo."""
    print("🚀 LSTM Siamese Text Similarity Demo")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("This demo shows how to use the LSTM Siamese system for text similarity matching.")
    print("It covers data extraction, model training, prediction, and Kubeflow deployment.")
    print()
    
    # Run demos
    demo_data_extraction()
    
    trained_successfully = demo_model_training()
    
    if trained_successfully:
        demo_kubeflow_pipeline()
    
    demo_command_line_usage()
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("="*60)
    
    # Ask about cleanup
    try:
        cleanup = input("\n❓ Clean up demo files? (y/n): ").lower().strip()
        if cleanup in ['y', 'yes']:
            cleanup_demo_files()
        else:
            print("💡 Demo files kept for your inspection")
    except KeyboardInterrupt:
        print("\n💡 Demo files kept")
    
    print(f"\n📚 For more information, see:")
    print(f"   - README_HIVE_LSTM_SIAMESE_SETUP.md")
    print(f"   - Hive_LSTM_Siamese_Testing_Notebook.ipynb")
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()