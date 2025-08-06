import os
import os
import numpy as np
import librosa
from scipy import signal
import math
import glob
from tqdm import tqdm

# Path to your folder with .wav files
folder_path = "/Scratch/repository/iahmad/harmonix-audio-trimmed"  # change this to your folder path

# Initialize array to store song IDs
song_ids = []

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        # Extract the part before '.wav'
        song_id = filename.rsplit(".wav", 1)[0]
        song_ids.append(song_id)

# Print or save the result
print("Extracted song IDs:", song_ids)






# === AUTHOR'S EXACT PARAMETERS ===
audio_sr = 40960  # Author's exact sample rate
window_size = 2048  # Author's FFT window
hop = 1024  # Author's hop length  
n_mels = 80  # Author's mel bands
fmin = 80  # Author's min frequency
fmax = 8000  # Author's max frequency (CRITICAL: not sr/2)
frame_size = 0.5  # Author's final frame size after downsampling
downsample = 20  # Author's downsampling factor
downsample_window = 21  # Author's median filter window

# === FUNCTION DICTIONARY (Author's exact mapping) ===
function_dict = {
    'intro': 0,
    'verse': 1, 
    'chorus': 2,
    'bridge': 3,
    'inst': 4,
    'outro': 5,
    'silence': 6,
}

def func_conversion(label):
    """Convert raw label to standardized function label (author's method)"""
    label = label.lower().strip()
    
    if label in function_dict:
        return label
    
    # ICASSP 2022 mapping (your method)
    substrings = [
        ("silence", "silence"), ("pre-chorus", "verse"), ("prechorus", "verse"),
        ("refrain", "chorus"), ("chorus", "chorus"), ("theme", "chorus"),
        ("stutter", "chorus"), ("verse", "verse"), ("rap", "verse"),
        ("section", "verse"), ("slow", "verse"), ("build", "verse"),
        ("dialog", "verse"), ("intro", "intro"), ("fadein", "intro"),
        ("opening", "intro"), ("bridge", "bridge"), ("trans", "bridge"),
        ("out", "outro"), ("coda", "outro"), ("ending", "outro"),
        ("break", "inst"), ("inst", "inst"), ("interlude", "inst"), 
        ("impro", "inst"), ("solo", "inst")
    ]
    
    for substr, mapped in substrings:
        if substr in label:
            return mapped
    
    return "inst"

# === MISSING AUGMENTATION METHODS (Your 10-file strategy) ===
# === NO AUGMENTATION VERSION ===
def create_augmentations():
    """
    Modified version: No augmentations, just original audio
    Returns only one augmentation method: 'original'
    """
    augmentations = ['original']  # Only original audio, no pitch shifts or pre-emphasis
    return augmentations

# === MISSING AUDIO_AUGMENTATION FUNCTION ===
# === SIMPLIFIED AUDIO_AUGMENTATION FUNCTION ===
def audio_augmentation(y, sr, method='original'):
    """
    Simplified version: No augmentations applied
    Always returns the original audio unchanged
    """
    if method == 'original':
        return y
    else:
        # If somehow a non-original method is passed, still return original
        print(f"Warning: {method} augmentation requested but augmentations are disabled. Returning original audio.")
        return y


# === AUTHOR'S EXACT FUNCTIONS ===
def median_downsample_feature_sequence(X, filt_len=21, down_sampling=20):
    """Author's exact downsampling method with median filtering"""
    assert filt_len % 2 == 1  # L needs to be odd
    filt_len = [1, filt_len]
    X_smooth = signal.medfilt2d(X, filt_len)
    X_smooth = X_smooth[:, ::down_sampling]
    return X_smooth

def get_functional_labels(annotations, n_frames, frame_size):
    """Author's exact label generation method (made robust)"""
    annotations = annotations.copy()
    
    # Remove annotations beyond frame limit
    valid_mask = np.floor(annotations['onset'] / frame_size) < n_frames
    annotations = annotations[valid_mask]
    
    if len(annotations) == 0:
        # Default to intro if no annotations
        boundary = np.zeros(shape=(n_frames,), dtype=np.int32)
        boundary[0] = 1
        function = np.zeros(shape=(n_frames,), dtype=np.int32)
        section = np.array(['intro'] * n_frames, dtype='U40')
        return boundary, function, section

    onset_in_frames = [math.floor(onset / frame_size) for onset in annotations['onset']]
    
    # Ensure frame 0 is included
    if 0 not in onset_in_frames:
        onset_in_frames.insert(0, 0)
        first_annotation = np.array([(0.0, annotations['section'][0])], 
                                   dtype=[('onset', np.float32), ('section', 'U40')])
        annotations = np.concatenate([first_annotation, annotations])
    
    onset_in_frames = sorted(set(onset_in_frames))
    
    # Create boundary array
    boundary = np.zeros(shape=(n_frames,), dtype=np.int32)
    boundary[onset_in_frames] = 1

    # Create section labels using cumulative sum (author's method)
    boundary_cumsum = np.cumsum(boundary) - 1
    
    # Handle edge cases
    max_boundary_idx = np.max(boundary_cumsum)
    if max_boundary_idx >= len(annotations):
        last_section = annotations['section'][-1]
        padding_needed = max_boundary_idx - len(annotations) + 1
        padding_annotations = np.array([(annotations['onset'][-1], last_section)] * padding_needed,
                                     dtype=[('onset', np.float32), ('section', 'U40')])
        annotations = np.concatenate([annotations, padding_annotations])
    
    section = np.take(annotations['section'], boundary_cumsum)
    function = np.array([function_dict[func_conversion(s)] for s in section])

    return boundary, function, section

def basename(path):
    """Helper function"""
    return os.path.basename(path)

# === CREATE ANNOTATION DICT FROM YOUR MAPPED FILES ===
def read_segment_annotation(annotation_dir, audio_files):
    """
    Create annotation_dict from your mapped annotation files
    Converts your format to author's expected format
    """
    annotation_dict = {}
    
    # Find all .npy annotation files
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.npy'))
    
    for ann_file in annotation_files:
        file_id = os.path.basename(ann_file).replace('.npy', '')
        
        try:
            # Load your mapped annotation data (shape: (n_segments, 2))
            segments = np.load(ann_file, allow_pickle=True)
            
            if len(segments) > 0 and segments.shape[1] >= 2:
                # Convert to author's format
                dtype = [('onset', np.float32), ('section', 'U40')]
                formatted_segments = []
                
                for i in range(len(segments)):
                    time_val = float(segments[i, 0])
                    label_val = str(segments[i, 1])
                    formatted_segments.append((time_val, label_val))
                
                if formatted_segments:
                    formatted_annotations = np.array(formatted_segments, dtype=dtype)
                    # Store in author's expected format: [annotator1, annotator2]
                    annotation_dict[file_id] = [formatted_annotations, None]
                    
        except Exception as e:
            print(f"Error processing annotation {file_id}: {e}")
            continue
    
    print(f"Loaded {len(annotation_dict)} annotations")
    return annotation_dict

# === AUTHOR'S EXACT create_feature_label FUNCTION (COMPLETED) ===
def create_feature_label(audio_files, annotation_dict, save_dir, downsample, downsample_window, max_len=935):
    """
    Author's exact create_feature_label function with all missing pieces filled in
    """
    
    # The missing augmentations list!
    augmentations = create_augmentations()
    print(f'Using {len(augmentations)} augmentation methods:')
    for i, aug in enumerate(augmentations):
        print(f'  {i+1}: {aug}')

    start = 0
    max_duration = 0
    total_processed = 0
    truncated_count = 0
    
    for i_f, file in enumerate(sorted(audio_files)[start:]):
        print(f'\nFile {i_f+start+1}/{len(audio_files)}: {basename(file)}')
        id = basename(file).replace('.wav', '').replace('.mp3', '')
        
        if id not in annotation_dict:
            print(f'⚠️ No annotation for {id}')
            continue
            
        multi_annotations = annotation_dict[id]

        # Load audio (author's method)
        try:
            y, sr = librosa.load(file, sr=audio_sr)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration > max_duration:
                max_duration = duration
        except Exception as e:
            print(f'⚠️ Error loading {file}: {e}')
            continue

        # Augment audio and get Labels (Author's exact structure!)
        for aug_method in augmentations:
            print(f'  Processing: {aug_method}')
            
            try:
                # The missing audio_augmentation call!
                y_aug = audio_augmentation(y, sr, method=aug_method)

                # Normalize amplitude (author's method)
                y_norm = (y_aug - y_aug.mean()) / y_aug.std()

                # Mel-spectrogram (author's exact parameters)
                spec = librosa.feature.melspectrogram(
                    y=y_norm, sr=sr, n_fft=window_size, hop_length=hop, 
                    n_mels=n_mels, fmin=fmin, fmax=fmax
                )
                spec = median_downsample_feature_sequence(spec, filt_len=downsample_window, down_sampling=downsample)

                # Chromagram (author's exact parameters)
                chromagram = librosa.feature.chroma_cqt(
                    y=y_norm, sr=sr, hop_length=hop, n_chroma=12
                )
                chromagram = median_downsample_feature_sequence(chromagram, filt_len=downsample_window, down_sampling=downsample)

                n_frames = spec.shape[1]
                print(f'    Generated {n_frames} frames')

                # Handle max_len constraint for transformer
                if n_frames > max_len:
                    print(f'    Truncating from {n_frames} to {max_len} frames')
                    spec = spec[:, :max_len]
                    chromagram = chromagram[:, :max_len]
                    n_frames = max_len
                    truncated_count += 1

                # The valid_len is the actual number of frames we're using
                # valid_len = n_frames
                print(f'    spec.shape: {spec.shape}')
                print(f'    chromagram.shape: {chromagram.shape}')
                print(f'    valid_len: {n_frames}')  # ADD THIS LINE

                # Convert annotations to labels (for each annotator)
                for annotator, annotations in enumerate(multi_annotations):
                    if annotations is not None:
                        # Filter annotations to max time
                        max_time = n_frames * frame_size
                        valid_annotations = annotations[annotations['onset'] <= max_time]
                        
                        boundary, function, section = get_functional_labels(
                            valid_annotations, n_frames=n_frames, frame_size=frame_size
                        )

                        print(f'    boundary.shape: {boundary.shape}')
                        print(f'    function.shape: {function.shape}')
                        print(f'    section.shape: {section.shape}')

                        # Check time dimensions (author's check)
                        shape_list = [
                            spec.shape[1],
                            chromagram.shape[1],
                            boundary.shape[0],
                            function.shape[0],
                            section.shape[0]
                        ]
                        assert shape_list.count(shape_list[0]) == len(shape_list), f"Shape mismatch: {shape_list}"

                        # Save processed data (author's exact format)
                        save_key = '_'.join([id, aug_method, 'a' + str(annotator+1)])
                        
                        with open(os.path.join(save_dir, save_key + '_drumspec.npy'), 'wb') as f:
                            np.save(f, np.transpose(spec, [1, 0]))  # [n_frames, 80]
                        with open(os.path.join(save_dir, save_key + '_drumchroma.npy'), 'wb') as f:
                            np.save(f, np.transpose(chromagram, [1, 0]))  # [n_frames, 12]
                        # with open(os.path.join(save_dir, save_key + '_boundary.npy'), 'wb') as f:
                        #     np.save(f, boundary)  # [n_frames]
                        # with open(os.path.join(save_dir, save_key + '_function.npy'), 'wb') as f:
                        #     np.save(f, function)  # [n_frames]
                        # with open(os.path.join(save_dir, save_key + '_section.npy'), 'wb') as f:
                        #     np.save(f, section)  # [n_frames]
                        # # Additional len file for training compatibility
                        # with open(os.path.join(save_dir, save_key + '_len.npy'), 'wb') as f:
                        #     np.save(f, n_frames)
                        
                        total_processed += 1
                        print(f'    ✅ Saved: {save_key}')
                        
            except Exception as e:
                print(f'    ❌ Error with {aug_method}: {e}')
                continue
    
    print(f'\nProcessing complete!')
    print(f'Max duration: {max_duration:.2f} seconds')
    print(f'Total samples processed: {total_processed}')
    print(f'Samples truncated: {truncated_count}')

# === DATASET CREATION ===
def load_and_combine_processed_files(processed_dir, output_file):
    """Load individual .npy files and create final .npz dataset"""
    
    spec_files = glob.glob(os.path.join(processed_dir, '*_spec.npy'))
    if not spec_files:
        raise ValueError(f"No processed files found in {processed_dir}")
    
    print(f"Found {len(spec_files)} processed samples")
    
    all_data = {
        'spec': [], 'chromagram': [], 'len': [],
        'boundary': [], 'function': [], 'section': []
    }
    
    for spec_file in sorted(spec_files):
        base_name = spec_file.replace('_spec.npy', '')
        
        try:
            # Load all components
            spec = np.load(spec_file)
            chroma = np.load(base_name + '_chroma.npy')
            boundary = np.load(base_name + '_boundary.npy')
            function = np.load(base_name + '_function.npy')
            section = np.load(base_name + '_section.npy')
            
            # Load or compute length
            len_file = base_name + '_len.npy'
            if os.path.exists(len_file):
                valid_len = int(np.load(len_file).item())
            else:
                valid_len = spec.shape[0]
                np.save(len_file, valid_len)
            
            # Store data
            all_data['spec'].append(spec)
            all_data['chromagram'].append(chroma)
            all_data['len'].append(valid_len)
            all_data['boundary'].append(boundary)
            all_data['function'].append(function)
            all_data['section'].append(section)
            
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            continue
    
    # Convert to numpy arrays
    for key in all_data.keys():
        if key != 'section':
            all_data[key] = np.array(all_data[key], dtype=object)
    
    # Save final dataset
    # np.savez_compressed(output_file, **all_data)
    print(f"✅ Saved final dataset: {output_file}")
    print(f"   Total samples: {len(all_data['spec'])}")
    print(f"   Sequence lengths: min={min(all_data['len'])}, max={max(all_data['len'])}, avg={np.mean(all_data['len']):.1f}")
    
    return all_data


# === MAIN FUNCTION WITH NO AUGMENTATION ===
def run_salami_preprocessing_no_aug(
    target_song_ids=None,
    audio_dir="/Scratch/repository/iahmad/harmonix-demucs/drums",
    annotation_dir="/Scratch/repository/iahmad/harmonix-mapped-annotation",
    output_dir="./harmonix-original-preprocessed-data",
    final_dataset_path="./harmonix_data/test_data.npz",
    max_len=935
):
    """
    Complete BEATLES preprocessing following author's exact method BUT WITHOUT AUGMENTATION
    """
    
    print("🎵 BEATLES Preprocessing - NO AUGMENTATION VERSION")
    print("=" * 70)
    print(f"🎯 Target songs: {len(target_song_ids) if target_song_ids else 'All'}")
    print(f"📏 Max sequence length: {max_len} frames ({max_len * frame_size:.1f}s)")
    print(f"🎼 Augmentation: DISABLED (1x original audio only)")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)
    
    # Step 1: Get audio files
    print(f"\n1️⃣ Getting audio files...")
    if target_song_ids:
        audio_files = []
        for song_id in target_song_ids:
            audio_path = os.path.join(audio_dir, f"{song_id}.wav")
            if os.path.exists(audio_path):
                audio_files.append(audio_path)
        print(f"   Found {len(audio_files)}/{len(target_song_ids)} target audio files")
    else:
        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        print(f"   Found {len(audio_files)} audio files")
    
    # Step 2: Create annotation dict
    print(f"\n2️⃣ Creating annotation dictionary...")
    annotation_dict = read_segment_annotation(annotation_dir, audio_files)
    
    # Step 3: Run preprocessing (NO augmentation!)
    print(f"\n3️⃣ Running preprocessing (NO AUGMENTATION)...")
    create_feature_label(
        audio_files=audio_files,
        annotation_dict=annotation_dict,
        save_dir=output_dir,
        downsample=downsample,
        downsample_window=downsample_window,
        max_len=max_len
    )
    
    # Step 4: Create final dataset
    print(f"\n4️⃣ Creating final dataset...")
    final_dataset = load_and_combine_processed_files(output_dir, final_dataset_path)
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"📊 Dataset: {final_dataset_path}")
    print(f"🔢 Total samples: {len(final_dataset['spec'])} (1x per song, no augmentation)")
    print(f"🚀 Ready for training!")
    
    return final_dataset_path

# === USAGE EXAMPLES ===
if __name__ == "__main__":
    
    # Your specific song IDs
    TARGET_SONG_IDS = song_ids
    # Option 1: Run with no augmentation
    dataset_path = run_salami_preprocessing_no_aug(
        target_song_ids=TARGET_SONG_IDS,
        max_len=935
    )
    
    print(f"\n✅ Final dataset ready: {dataset_path}")
    print("🎯 Expected behavior: 1 sample per song (vs 10 samples with augmentation)")

# === COMPARISON: BEFORE vs AFTER ===
print("""
📊 COMPARISON - Before vs After:

BEFORE (With Augmentation):
- 10 augmentation methods (5 pitch shifts × 2 pre-emphasis values)
- 132 songs × 10 augmentations = 1,320 samples
- Files: song_id_pitch_X_preemph_Y_a1_*.npy

AFTER (No Augmentation):
- 1 method: 'original' only
- 132 songs × 1 original = 132 samples  
- Files: song_id_original_a1_*.npy

RESULT:
- 10x faster preprocessing
- 10x smaller dataset size
- No synthetic data, only original audio
- May need longer training or different strategies for good performance
""")



