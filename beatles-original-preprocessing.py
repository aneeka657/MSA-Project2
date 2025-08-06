import os
import os
import numpy as np
import librosa
from scipy import signal
import math
import glob
from tqdm import tqdm

# # Path to your folder with .wav files
# folder_path = "/Scratch/repository/iahmad/beatles"  # change this to your folder path

# # Initialize array to store song IDs
# song_ids = []

# # Loop through files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".wav"):
#         # Extract the part before '.wav'
#         song_id = filename.rsplit(".wav", 1)[0]
#         song_ids.append(song_id)

# # Print or save the result
# print("Extracted song IDs:", song_ids)




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
                        
                        with open(os.path.join(save_dir, save_key + '_spec.npy'), 'wb') as f:
                            np.save(f, np.transpose(spec, [1, 0]))  # [n_frames, 80]
                        with open(os.path.join(save_dir, save_key + '_chroma.npy'), 'wb') as f:
                            np.save(f, np.transpose(chromagram, [1, 0]))  # [n_frames, 12]
                        with open(os.path.join(save_dir, save_key + '_boundary.npy'), 'wb') as f:
                            np.save(f, boundary)  # [n_frames]
                        with open(os.path.join(save_dir, save_key + '_function.npy'), 'wb') as f:
                            np.save(f, function)  # [n_frames]
                        with open(os.path.join(save_dir, save_key + '_section.npy'), 'wb') as f:
                            np.save(f, section)  # [n_frames]
                        # Additional len file for training compatibility
                        with open(os.path.join(save_dir, save_key + '_len.npy'), 'wb') as f:
                            np.save(f, n_frames)
                        
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
    np.savez_compressed(output_file, **all_data)
    print(f"✅ Saved final dataset: {output_file}")
    print(f"   Total samples: {len(all_data['spec'])}")
    print(f"   Sequence lengths: min={min(all_data['len'])}, max={max(all_data['len'])}, avg={np.mean(all_data['len']):.1f}")
    
    return all_data


# === MAIN FUNCTION WITH NO AUGMENTATION ===
def run_salami_preprocessing_no_aug(
    target_song_ids=None,
    audio_dir="/Scratch/repository/iahmad/beatles",
    annotation_dir="/Scratch/repository/iahmad/beatles-mapped-annotation",
    output_dir="./beatles-original-preprocessed-data",
    final_dataset_path="./beatles_data/test_data.npz",
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
    # TARGET_SONG_IDS = ['15_-_Carry_That_Weight', '14_-_Run_For_Your_Life', '07_-_Cant_Buy_Me_Love', '01_-_It_Wont_Be_Long', '02_-_Im_a_Loser', '03_-_Anna_(Go_To_Him)', '03_-_Mother_Natures_Son', '04_-_Nowhere_Man', '15_-_Why_Dont_We_Do_It_In_The_Road', '10_-_Honey_Dont', '11_-_Doctor_Robert', '03_-_Across_the_Universe', '04_-_Chains', '02_-_I_Should_Have_Known_Better', '04_-_Dont_Bother_Me', '10_-_Im_So_Tired', '03_-_Youve_Got_To_Hide_Your_Love_Away', '05_-_Dig_It', '14_-_Tomorrow_Never_Knows', '12_-_Sgt._Peppers_Lonely_Hearts_Club_Band_(Reprise)', '09_-_Honey_Pie', '02_-_Yer_Blues', '13_-_She_Came_In_Through_The_Bathroom_Window', '04_-_Blue_Jay_Way', '08_-_Love_Me_Do', '09_-_And_Your_Bird_Can_Sing', '17_-_Her_Majesty', '11_-_Black_Bird', '13_-_What_Youre_Doing', '01_-_Help!', '04_-_Everybodys_Got_Something_To_Hide_Except_Me_and_My_Monkey', '09_-_P._S._I_Love_You', '04_-_Getting_Better', '02_-_The_Fool_On_The_Hill', '01_-_Drive_My_Car', '04_-_Love_You_To', '02_-_With_A_Little_Help_From_My_Friends', '03_-_If_I_Fell', '12_-_You_Cant_Do_That', '08_-_Good_Day_Sunshine', '14_-_Dizzy_Miss_Lizzy', '10_-_Things_We_Said_Today', '06_-_Shes_Leaving_Home', '11_-_Good_Morning_Good_Morning', '12_-_A_Taste_Of_Honey', '13_-_Got_To_Get_You_Into_My_Life', '05_-_Here,_There_And_Everywhere', '09_-_Martha_My_Dear', '16_-_I_Will', '04_-_Oh!_Darling', '09_-_One_After_909', '10_-_Baby_Its_You', '07_-_Hello_Goodbye', '13_-_Theres_A_Place', '06_-_Youre_Going_To_Lose_That_Girl', '01_-_Sgt._Peppers_Lonely_Hearts_Club_Band', '01_-_A_Hard_Days_Night', '05_-_Octopuss_Garden', '10_-_Savoy_Truffle', '17_-_Julia', '03_-_Maxwells_Silver_Hammer', '05_-_And_I_Love_Her', '09_-_Girl', '09_-_You_Never_Give_Me_Your_Money', '09_-_When_Im_Sixty-Four', '12_-_Piggies', '12_-_Get_Back', '11_-_In_My_Life', '13_-_Good_Night', '10_-_Baby_Youre_A_Rich_Man', '14_-_Twist_And_Shout', '11_-_Every_Little_Thing', '13_-_A_Day_In_The_Life', '06_-_Yellow_Submarine', '01_-_Taxman', '04_-_Im_Happy_Just_To_Dance_With_You', '05_-_Your_Mother_Should_Know', '02_-_Something', '08_-_Eight_Days_a_Week', '12_-_Polythene_Pam', '07_-_Please_Mister_Postman', '14_-_Everybodys_Trying_to_Be_My_Baby', '03_-_Glass_Onion', '02_-_Norwegian_Wood_(This_Bird_Has_Flown)', '06_-_Mr._Moonlight', '10_-_Sun_King', '16_-_The_End', '08_-_Strawberry_Fields_Forever', '07_-_While_My_Guitar_Gently_Weeps', '02_-_Eleanor_Rigby', '07_-_Ticket_To_Ride', '01_-_Back_in_the_USSR', '12_-_I_Dont_Want_to_Spoil_the_Party', '02_-_The_Night_Before', '07_-_Long_Long_Long', '10_-_Im_Looking_Through_You', '01_-_Come_Together', '05_-_Ill_Follow_the_Sun', '07_-_Please_Please_Me', '07_-_She_Said_She_Said', '12_-_I_Want_To_Tell_You', '05_-_Fixing_A_Hole', '14_-_Dont_Pass_Me_By', '01_-_Two_of_Us', '08_-_Any_Time_At_All', '08_-_What_Goes_On', '10_-_You_Really_Got_A_Hold_On_Me', '06_-_Ask_Me_Why', '10_-_For_No_One', '10_-_The_Long_and_Winding_Road', '12_-_Ive_Just_Seen_a_Face', '09_-_Hold_Me_Tight', '07_-_Maggie_Mae', '02_-_All_Ive_Got_To_Do', '05_-_Little_Child', '04_-_I_Me_Mine', '12_-_Devil_In_Her_Heart', '04_-_Rock_and_Roll_Music', '08_-_Roll_Over_Beethoven', '03_-_Babys_In_Black', '09_-_Ill_Cry_Instead', '06_-_The_Continuing_Story_of_Bungalow_Bill', '11_-_All_You_Need_Is_Love', '10_-_Lovely_Rita', '14_-_Money', '13_-_Rocky_Raccoon', '03_-_Flying', '06_-_I_Am_The_Walrus', '06_-_I_Want_You', '08_-_Act_Naturally', '08_-_Revolution_1', '02_-_Dig_a_Pony', '08_-_Happiness_is_a_Warm_Gun', '04_-_Ob-La-Di,_Ob-La-Da', '07_-_Here_Comes_The_Sun', '03_-_You_Wont_See_Me', '06_-_The_Word', '08_-_Ive_Got_A_Feeling', '03_-_Im_Only_Sleeping', '11_-_When_I_Get_Home', '01_-_Birthday', '07_-_Michelle', '09_-_Its_Only_Love', '12_-_Revolution_9', '05_-_Wild_Honey_Pie', '13_-_Not_A_Second_Time', '01_-_No_Reply', '11_-_I_Wanna_Be_Your_Man', '02_-_Dear_Prudence', '12_-_Wait', '05_-_Think_For_Yourself', '02_-_Misery', '08_-_Within_You_Without_You', '09_-_Penny_Lane', '10_-_You_Like_Me_Too_Much', '11_-_Cry_Baby_Cry', '09_-_Words_of_Love', '14_-_Golden_Slumbers', '01_-_Magical_Mystery_Tour', '03_-_Lucy_In_The_Sky_With_Diamonds', '06_-_Helter_Skelter', '08_-_Because', '03_-_All_My_Loving', '13_-_If_I_Needed_Someone', '11_-_Tell_Me_What_You_See', '07_-_Kansas_City-_Hey,_Hey,_Hey,_Hey', '05_-_Another_Girl', '11_-_Mean_Mr_Mustard', '06_-_Till_There_Was_You', '11_-_For_You_Blue', '05_-_Sexy_Sadie', '07_-_Being_For_The_Benefit_Of_Mr._Kite!', '05_-_Boys', '01_-_I_Saw_Her_Standing_There', '11_-_Do_You_Want_To_Know_A_Secret', '04_-_I_Need_You', '06_-_Let_It_Be', '13_-_Yesterday', '13_-_Ill_Be_Back', '06_-_Tell_Me_Why']
    TARGET_SONG_IDS = [
    "01_-_It_Wont_Be_Long",
    "01_-_Sgt._Peppers_Lonely_Hearts_Club_Band",
    "02_-_All_Ive_Got_To_Do",
    "02_-_Im_a_Loser",
    "03_-_Babys_In_Black",
    "03_-_Im_Only_Sleeping",
    "03_-_Maxwells_Silver_Hammer",
    "03_-_You_Wont_See_Me",
    "03_-_Youve_Got_To_Hide_Your_Love_Away",
    "04_-_Dont_Bother_Me",
    "04_-_Everybodys_Got_Something_To_Hide_Except_Me_and_My_Monkey",
    "04_-_Im_Happy_Just_To_Dance_With_You",
    "05_-_Ill_Follow_the_Sun",
    "05_-_Octopuss_Garden",
    "05_-_Wild_Honey_Pie",
    "06_-_Helter_Skelter",
    "06_-_Youre_Going_To_Lose_That_Girl",
    "07_-_Cant_Buy_Me_Love",
    "08_-_Happiness_is_a_Warm_Gun",
    "08_-_Ive_Got_A_Feeling",
    "09_-_Ill_Cry_Instead",
    "09_-_Its_Only_Love",
    "09_-_When_Im_Sixty-Four",
    "09_-_You_Never_Give_Me_Your_Money",
    "10_-_Baby_Its_You",
    "10_-_Baby_Youre_A_Rich_Man",
    "10_-_Honey_Dont",
    "12_-_Ive_Just_Seen_a_Face",
    "12_-_Revolution_9",
    "13_-_Ill_Be_Back",
    "13_-_Theres_A_Place",
    "14_-_Dont_Pass_Me_By",
    "14_-_Everybodys_Trying_to_Be_My_Baby",
    "15_-_Why_Dont_We_Do_It_In_The_Road",
    "16_-_The_End"
]

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



