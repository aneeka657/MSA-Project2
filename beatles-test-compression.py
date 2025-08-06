import numpy as np
import glob
import os

def create_dataset_original_style(
    processed_dir="/Scratch/repository/msa/MSATSUNGPING/harmonix-aug-preprocessed-data",
    output_file="./harmonix_data/train_data.npz"
):
    
    """
    
    print(f"📁 Loading processed files from: {processed_dir}")
    print(f"💾 Will save to: {output_file}")
    print(f"📐 Strategy: Original author's method (no saved lengths)")
    
    # Find all spec files
    spec_files = glob.glob(os.path.join(processed_dir, '*_spec.npy'))
    if not spec_files:
        raise ValueError(f"No processed files found in {processed_dir}")
    
    print(f"🔍 Found {len(spec_files)} processed samples")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Data containers - following original author's structure
    all_data = {
        'spec': [],
        'chromagram': [], 
        'boundary': [],
        'function': [],
        'section': []
        # NOTE: No 'len' key - original author didn't save lengths!
    }
    
    successful_loads = 0
    sequence_lengths = []
    
    for spec_file in sorted(spec_files):
        base_name = spec_file.replace('_spec.npy', '')
        sample_name = os.path.basename(base_name)
        
        try:
            # Load the 5 files that the original author saved
            spec = np.load(spec_file)                    # [n_frames, 80]
            chroma = np.load(base_name + '_chroma.npy')  # [n_frames, 12]  
            boundary = np.load(base_name + '_boundary.npy') # [n_frames]
            function = np.load(base_name + '_function.npy') # [n_frames]
            section = np.load(base_name + '_section.npy')   # [n_frames]
            
            # Verify shapes are consistent
            n_frames = spec.shape[0]
            if (chroma.shape[0] != n_frames or 
                boundary.shape[0] != n_frames or 
                function.shape[0] != n_frames or 
                section.shape[0] != n_frames):
                print(f"⚠️ Shape mismatch in {sample_name}, skipping")
                continue
            
            sequence_lengths.append(n_frames)
            
            # Store data exactly as original author did
            all_data['spec'].append(spec)
            all_data['chromagram'].append(chroma)
            all_data['boundary'].append(boundary)
            all_data['function'].append(function)
            all_data['section'].append(section)
            
            successful_loads += 1
            if successful_loads <= 5:
                print(f"✅ Loaded {sample_name}: n_frames={n_frames}")
            elif successful_loads == 6:
                print("   ... (continuing silently)")
                
        except Exception as e:
            print(f"❌ Error loading {sample_name}: {e}")
            continue
    
    print(f"\n📊 Loading Summary:")
    print(f"   ✅ Successful: {successful_loads}")
    print(f"   📏 Sequence lengths: min={min(sequence_lengths)}, max={max(sequence_lengths)}, avg={np.mean(sequence_lengths):.1f}")
    
    if successful_loads == 0:
        raise ValueError("No files were successfully loaded!")
    
    # Convert to object arrays (original author's approach for variable lengths)
    print(f"📦 Converting to object arrays (original author's method)...")
    final_data = {}
    
    for key in all_data.keys():
        if key == 'section':
            # String arrays need object type
            final_data[key] = np.array(all_data[key], dtype=object)
        else:
            # Numeric arrays with variable lengths need object type
            final_data[key] = np.array(all_data[key], dtype=object)
    
    # Save dataset
    print(f"💾 Saving dataset...")
    try:
        np.savez_compressed(output_file, **final_data)
        print(f"✅ Successfully saved: {output_file}")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        raise
    
    # Print final info
    print(f"\n📊 Final Dataset (Original Author's Style):")
    print(f"   📁 File: {output_file}")
    print(f"   🔢 Total samples: {len(final_data['spec'])}")
    print(f"   📋 Keys: {list(final_data.keys())}")
    
    for key, array in final_data.items():
        print(f"   📋 {key}: shape={array.shape}, dtype={array.dtype}")
        if len(array) > 0:
            first_shape = array[0].shape if hasattr(array[0], 'shape') else 'scalar'
            print(f"       First element: {first_shape}")
    
    return final_data

def create_generator_original_style(data):
    """
    Generator following original author's approach:
    - No saved valid_len 
    - Calculate length from actual sequence shape
    - This is how they handled variable lengths in training
    """
    for spec, chromagram, boundary, function, section in \
            zip(data['spec'], data['chromagram'], 
                data['boundary'], data['function'], data['section']):
        
        # Calculate valid_len from actual sequence length (original author's method)
        valid_len = spec.shape[0]  # This is what they do in training!
        
        yield spec, chromagram, valid_len, boundary, function, section

def verify_original_style_dataset(dataset_path):
    """Verify dataset works with original author's training approach"""
    
    print(f"\n🔍 Verifying dataset (original style): {dataset_path}")
    
    try:
        data = np.load(dataset_path, allow_pickle=True)
        
        print("✅ Dataset loaded successfully!")
        print("Contents:")
        for key in sorted(data.files):
            array = data[key]
            print(f"  {key}: shape={array.shape}, dtype={array.dtype}")
        
        # Test generator (original author's way)
        print("\n🧪 Testing original author's generator approach:")
        gen = create_generator_original_style(data)
        
        for i, (spec, chromagram, valid_len, boundary, function, section) in enumerate(gen):
            print(f"Sample {i+1}:")
            print(f"  spec: {spec.shape}, dtype: {spec.dtype}")
            print(f"  chromagram: {chromagram.shape}, dtype: {chromagram.dtype}")  
            print(f"  valid_len: {valid_len} (calculated from spec.shape[0])")
            print(f"  boundary: {boundary.shape}, dtype: {boundary.dtype}")
            print(f"  function: {function.shape}, dtype: {function.dtype}")
            print(f"  section: {section.shape}, dtype: {section.dtype}")
            
            if i >= 1:  # Test first 2 samples
                break
        
        print("✅ Original style verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

# === COMPARISON: Original vs Your Modified Approach ===
def show_comparison():
    print("""
📊 COMPARISON: Original Author vs Your Modified Approach

ORIGINAL AUTHOR'S METHOD:
✅ Files saved: _spec.npy, _chroma.npy, _boundary.npy, _function.npy, _section.npy
❌ NO _len.npy files saved
✅ Variable-length sequences stored as object arrays
✅ valid_len calculated in training: valid_len = spec.shape[0]
✅ tf.data.Dataset.padded_batch() handles dynamic padding

YOUR MODIFIED METHOD:
✅ Files saved: same 5 + _len.npy  
✅ valid_len explicitly saved and loaded
✅ Can use either fixed-length padding or variable-length
✅ More explicit control over sequence lengths

BOTH WORK! The training code handles both approaches because:
- It uses valid_len for masking regardless of source
- tf.data.Dataset.padded_batch() handles variable lengths
- Sequence masking prevents issues with padding
""")

# === USAGE ===
if __name__ == "__main__":
    
    print("🎯 Creating dataset following ORIGINAL author's approach...")
    
    # Create dataset exactly like original author
    dataset = create_dataset_original_style(
        processed_dir="./harmonix-aug-preprocessed-data",
        output_file="./harmonix_data/train_data.npz"
    )
    
    # Verify it works
    verify_original_style_dataset("./harmonix_data/train_data.npz")
    
    # Show comparison
    show_comparison()
    
    print("\n🎉 Dataset created following original author's method!")
    print("🚀 This will work with the training code!")
    print("📝 The training code will calculate valid_len = spec.shape[0] automatically")
    
    # Example of how to use in training (original author's way):
    print("""
📋 Usage in Training (Original Author's Way):

def generator(data):
    for spec, chromagram, boundary, function, section in \\
            zip(data['spec'], data['chromagram'], 
                data['boundary'], data['function'], data['section']):
        
        valid_len = spec.shape[0]  # Calculate length from shape!
        yield spec, chromagram, valid_len, boundary, function, section

# Then use tf.data.Dataset.padded_batch() for dynamic padding
""")