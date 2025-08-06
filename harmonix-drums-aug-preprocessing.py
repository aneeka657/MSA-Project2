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
song_ids.sort()
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
def create_augmentations():
    """
    Create the missing augmentations list for author's preprocessing.py
    Your strategy: Pre-emphasis {0.7, 0.97} applied to pitch shifts [-2,-1,0,1,2] = 10 files
    """
    augmentations = []
    pitch_shifts = [-2, -1, 0, 1, 2]
    preemph_coeffs = [0.7, 0.97]
    
    for n_steps in pitch_shifts:
        for alpha in preemph_coeffs:
            augmentations.append(f'pitch_{n_steps}_preemph_{alpha:.2f}')
    
    return augmentations

# === MISSING AUDIO_AUGMENTATION FUNCTION ===
def audio_augmentation(y, sr, method='original'):
    """
    The missing audio_augmentation function from author's preprocessing.py
    Applies pitch shift + pre-emphasis as described in the paper
    """
    if method == 'original':
        return y
    
    y_aug = y.copy()
    
    if method.startswith('pitch_') and '_preemph_' in method:
        try:
            # Parse: 'pitch_-2_preemph_0.70' -> n_steps=-2, alpha=0.70
            parts = method.split('_')
            n_steps = int(parts[1])
            alpha = float(parts[3])
            
            # Step 1: Pitch shift (author's exact method)
            if n_steps != 0:
                y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=n_steps)
            
            # Step 2: Pre-emphasis (author's exact method)
            if alpha > 0:
                y_preemph = np.zeros_like(y_aug)
                y_preemph[0] = y_aug[0]  # First sample unchanged
                y_preemph[1:] = y_aug[1:] - alpha * y_aug[:-1]  # Pre-emphasis filter
                y_aug = y_preemph
                
        except Exception as e:
            print(f"Error in audio_augmentation: {e}")
            return y
    
    return y_aug

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

# === MAIN FUNCTION ===
def run_salami_preprocessing(
    target_song_ids=None,
    audio_dir="/Scratch/repository/iahmad/harmonix-demucs/drums",
    annotation_dir="/Scratch/repository/iahmad/harmonix-mapped-annotation",
    output_dir="./harmonix-aug-preprocessed-data",
    final_dataset_path="./harmonix_data/train_data.npz",
    max_len=935
):
    """
    Complete SALAMI preprocessing following author's exact method
    """
    
    print("🎵 SALAMI Preprocessing - Author's Exact Method + Your Requirements")
    print("=" * 70)
    print(f"🎯 Target songs: {len(target_song_ids) if target_song_ids else 'All'}")
    print(f"📏 Max sequence length: {max_len} frames ({max_len * frame_size:.1f}s)")
    print(f"🎼 Augmentation: 10x (pitch shifts + pre-emphasis)")
    
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
    
    # Step 3: Run preprocessing (author's exact method!)
    print(f"\n3️⃣ Running preprocessing...")
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
    print(f"🚀 Ready for training!")
    
    return final_dataset_path

# === USAGE ===
if __name__ == "__main__":
    
    # Your specific song IDs
    TARGET_SONG_IDS = [ '0220_promiscuous', '0223_reallove', '0227_replay', '0229_ridingonthewind', '0231_rightthurr', '0233_rocknrollnightmare', '0234_roundandround', '0235_rudeboy', '0236_rumpshaker', '0237_run', '0239_sandm', '0240_sandstorm', '0241_satellite', '0242_satisfaction', '0243_saucyjack', '0244_sayaah', '0246_scenario', '0247_scream', '0248_screamingfor', '0249_sensualseduction', '0250_sexyandiknowit', '0251_sexychick', '0254_sixsixsix', '0255_sleepwalker', '0257_somebodytolove', '0258_sorry', '0259_spiceupyourlife', '0261_starships', '0264_stereolove', '0265_straightlines', '0266_straightup', '0267_superfreak', '0269_sweettalk', '0270_takeovercontrol', '0272_teachmehowtojerk', '0273_technologic', '0274_temperature', '0276_texasflood', '0278_thatsthahomie', '0279_thebreaks', '0280_thedayitriedtolive', '0281_theedgeofglory', '0282_thegreatsatan', '0283_thehumptydance', '0284_thehustle', '0285_thekill', '0286_theragelive', '0287_therewasatime', '0288_thewayiare', '0289_thisisexile', '0290_thisishowwedoit', '0291_thrasher', '0292_timeslikethese', '0293_tomsawyer2', '0294_tonighttonight', '0295_toxic', '0296_transmaniaconmc', '0297_troublecomesrunning', '0298_turnmeon', '0299_turnthebeataround', '0301_unclejohnsband', '0302_venus', '0303_wakeupdead', '0304_weaponofchoice', '0305_wenospeakamericano', '0306_werunthis', '0307_whatdoesntkillyou', '0308_whatislove', '0309_whatsmyname', '0310_whineup', '0312_whitefalconfuzz', '0314_whoomp', '0315_whygo', '0316_wildones', '0318_windup', '0321_wonderwall', '0322_wordup', '0323_yeah', '0324_yeah3x', '0325_ymca', '0326_yougotittherightstuff', '0327_youmakemefeel', '0328_youmakemefeeldc', '0329_youreajerk', '0330_50waystosaygoodbye', '0331_againstallodds', '0332_alejandro', '0333_allday', '0334_alltherightplaces', '0335_allyourlife', '0336_alonewithyou', '0337_alwayssomethingthere', '0338_animal', '0339_aslongasyouloveme', '0340_backintime', '0341_baggageclaim', '0342_bangpop', '0343_banjo', '0344_beautifullife', '0345_beinlovetonight', '0346_birthdaydress', '0347_bitch', '0348_blackout', '0349_blow', '0350_blowme', '0351_blowup', '0352_boogiewonderland', '0353_bottomsup', '0354_brandnewday', '0355_brokenhearted', '0356_californiagurls', '0357_cheers', '0359_cominghome', '0360_coolerthanme', '0361_countrymustbecountrywide', '0362_crazygirl', '0363_darkside', '0364_deuces', '0365_dirtroadanthem', '0366_domino', '0367_dontdreamitsover', '0368_dontwakemeup', '0370_downonme', '0371_downunder', '0372_drinkinmyhand', '0373_driveby', '0374_drunkonyou', '0375_dynamite', '0376_earthquake', '0377_eatdirt', '0379_et', '0380_everybodytalks', '0381_everythingyouwant', '0382_fatherofmine', '0383_feelsoclose', '0384_finally', '0385_fiveoclock', '0387_free', '0388_gangnamstyle', '0389_gettinoveryou', '0390_getyourselfbackhome', '0391_givemeallyourluvin', '0392_givemeyourhand', '0393_giveyourheartabreak', '0394_godgavemetoyou', '0395_gold', '0396_goodfeeling', '0397_goodlife', '0398_goodtime', '0399_gotmygirls', '0400_heatofthemoment', '0401_hello', '0402_herestous', '0404_holditagainstme', '0405_home', '0406_honeybee', '0407_humpinaround', '0408_icanonlyimagine', '0409_idontlikeyou', '0410_idontwantthisnighttoend', '0411_ifihadyou', '0412_ifitmakesyouhappy', '0413_ifyoucouldonlysee', '0414_illbe', '0416_impossible', '0417_imtoosexy', '0418_inthedark', '0419_itgirl', '0420_itsrainingmen', '0421_ittakestwo', '0422_itwillrain', '0423_iwannago', '0424_justadream', '0425_justakiss', '0426_justcantgetenough', '0427_justthewayyouare', '0428_kingofanything', '0429_kissinu', '0430_lastfridaynight', '0431_letitrain', '0432_letsgo', '0433_lighters', '0434_lights', '0435_lookatmenow', '0436_loveactually', '0437_lovelikewoe', '0438_loverlover', '0439_lovethewayyoulie', '0440_loveyoulikealovesong', '0442_lullaby', '0443_magic', '0444_maybe', '0445_memories', '0446_midnightcity', '0447_mindyourmanners', '0448_mine', '0449_misery', '0450_more', '0451_mrknowitall', '0452_myfirstkiss', '0453_mygirl', '0454_myohmy', '0455_nevertearusapart', '0456_nohands', '0457_nothing', '0458_notorious', '0459_notoveryou', '0460_numb', '0461_obsession', '0462_oldalabama', '0463_onemorenight', '0465_onething', '0466_onthedarkside', '0467_onthefloor', '0468_openarms', '0470_ours', '0471_paradise', '0472_partofme', '0473_payphone', '0474_pleasedontgo', '0475_poisonedwithlove', '0476_pontoon', '0477_poundthealarm', '0478_prayforyou', '0479_pricetag', '0480_raiseyourglass', '0481_reallove', '0482_redsolocup', '0483_ridinsolo', '0484_rockthatbody', '0485_rollinginthedeep', '0487_secrets', '0488_september', '0489_shakeyourbody', '0490_showme', '0491_skyscraper', '0492_sm', '0494_somebodythatiusedtoknow', '0495_someonelikeyou', '0496_sparksfly', '0497_speakers', '0498_springsteen', '0499_starbuckssmile', '0500_stay', '0501_stayawhile', '0502_stronger', '0503_stucklikeglue', '0504_suspiciousminds', '0505_takeitoff', '0506_takemehometonight', '0507_teenagedream', '0508_tennesseeme', '0509_thelazysong', '0510_theonethatgotaway', '0511_theonlyexception', '0512_theshowgoeson', '0513_thetime', '0514_thetimeofmylife', '0515_thisafternoon', '0517_tilltheworldends', '0518_timeaftertime', '0519_titanium', '0520_tonight', '0521_tooclose', '0522_turnupthemusic', '0523_ucanttouchthis', '0524_undoit', '0525_venus', '0526_waitingoutsidethelines', '0527_wantuback', '0528_werunthenight', '0529_werwhower', '0530_whatthehell', '0531_whenwestandtogether', '0532_whistle', '0533_whosays', '0534_wideawake', '0535_withoutyou', '0536_workout', '0537_writteninthestars', '0538_you', '0539_youandi', '0540_youandme', '0541_youdroppedabombonme', '0542_yougetwhatyougive', '0543_yougonnafly', '0544_yougottabe', '0545_youkeepmehangingon', '0546_18days', '0548_2getherextended', '0550_aboutyounow', '0551_accordingtoyou', '0552_addicted', '0554_allalright', '0556_allaroundtheworld', '0558_allnightlong', '0559_alltherightmoves', '0560_almostlovenow', '0561_alorsondanse', '0562_alreadygone', '0564_americanboybenny', '0565_americanhoney', '0566_americanwoman', '0567_anythingcouldhappen', '0568_apologize', '0569_arrow', '0570_aslongasyouloveme', '0571_atlantis', '0572_awake', '0573_babyludacris', '0574_badromanceappboyz', '0575_badromanceskrillexremix', '0576_barbrastreisand', '0577_battlefieldmainversion', '0578_bealright', '0579_beatitstudioversion', '0580_beautiful', '0582_beautifulday', '0583_beautifulgirls', '0585_beautyandabeat', '0586_bed', '0587_bedrock', '0588_believe', '0589_bestieverhad', '0590_betterintime', '0591_biggirlsdontcry', '0592_blameit', '0594_bleedinglove', '0595_blonde', '0597_boomboomguetta', '0598_boomboompow', '0599_boyfriend', '0600_breakeven', '0602_breaktheice', '0603_breaktheicejasonnevinsmix', '0604_bustitbabypart2', '0607_cantholdus', '0608_canttellmenothing', '0609_carryon', '0610_castle', '0611_catchingfeelings', '0614_cinema', '0615_circus', '0616_citygirl', '0617_citylights', '0618_clocks', '0619_closer', '0620_closertands', '0621_clubcanthandleme', '0622_clumsy', '0623_comeongethigher', '0625_complicated', '0626_coolerthanme', '0627_cowboyboots', '0628_crankthatsouljaboy', '0629_crush', '0630_crushcrushcrush', '0631_curiosity', '0632_cyclone', '0633_dancingmachine', '0634_dangerous', '0635_daynnite', '0636_daynnitecrookersremix', '0637_daynniteremix', '0638_deadandgone', '0639_dieinyourarms', '0640_disturbia2', '0642_dontconfess', '0643_dontrush', '0644_dontsayaword', '0645_dontstopthemusic', '0647_dontthinkidontthinkaboutit', '0648_donttrustme', '0649_down', '0650_downlilwayne', '0651_doyouremember', '0652_drovemewild', '0653_dynamite', '0654_dynamitemixinmarc', '0655_earthquakey', '0657_enursbonfire', '0659_everybody', '0661_everythingaboutyou', '0662_everytimewetouch', '0663_explosions', '0664_eyesextendedmix', '0665_fall', '0666_fallin', '0668_fallinganthem', '0669_fastforwardffwd', '0670_fearless', '0671_feedback', '0672_feelitinmybones', '0673_feelsliketonight', '0676_figure8', '0677_fireburning', '0678_fireflies', '0679_firsttime', '0680_fixaheart', '0681_flashinglights', '0682_forever', '0683_foreverextendedmix', '0685_forgottolaugh', '0686_fortheloveofadaughter', '0687_getouttamyway', '0688_getthepartystarted', '0689_gettinoveryou', '0690_givemeeverything', '0691_givesyouhell', '0692_giveyourheartabreakdos', '0693_gold', '0695_goodbyegoodbye', '0696_goodgirlsgobad', '0697_goodlife', '0698_goodtime', '0699_gotmoney', '0700_gottabesomebody', '0701_gottabeyou', '0702_greenlight', '0704_guitarstringweddingring', '0705_haditall', '0706_halcyon', '0707_halfwaygone', '0708_haloradioedit', '0709_hangingon', '0710_hardjeezy', '0712_heartless', '0713_heartofgoldnow', '0714_hell', '0715_hellodadalife', '0716_heroheroine', '0717_heysoulsister', '0718_heytheredelilah', '0719_hipsdontlie', '0721_holdup', '0723_hotelroomservice', '0724_hotmessdj', '0725_hotncold', '0726_hotplay', '0727_howcomeyoudontwantme', '0728_howyouremindme', '0729_hurricane', '0730_hurtssogood', '0731_icouldntbeyourfriend', '0732_idontwannabeinlove', '0733_ifihadyou', '0734_iftodaywasyourlastday', '0735_ifuseeamy', '0736_igotafeeling', '0737_ihatethispart', '0738_ihearnoises', '0739_ikissedagirl', '0740_ikissedagirljasonnevins', '0741_iknowyoucare', '0742_iknowyouwantme2', '0743_iminloveiwannadoit', '0744_immabe', '0745_immabewolf', '0746_imnotyourhero', '0748_imyours', '0749_indestructible2', '0750_ineedyourlove', '0751_inlovewithagirl', '0752_inmyhead', '0753_inreallife', '0756_istillmissyou', '0758_itgetsbetter', '0759_itgirljasonnevins', '0760_iwannarock', '0761_iwant', '0762_iwantitthatway', '0763_iwasafool', '0764_iwish', '0765_jimmyiovine', '0766_justdance', '0767_justfine', '0768_keepsgettinbetter', '0769_killa', '0770_kingofmycastle', '0771_kisskissmainversion', '0773_kissmethruthephone', '0774_knockyoudown', '0775_knockyoudownbimbojones', '0776_krazy', '0777_lebump', '0778_letitrock', '0781_lettinggoduttylove', '0782_lifeafteryou', '0783_lighton', '0784_lightweight', '0785_littleliesnow', '0786_livelikeweredying', '0787_liveyourlife', '0789_livingroom', '0790_locapeople', '0791_lollipop', '0792_lovedrunk', '0793_lovegame', '0794_lovegamedave', '0798_lovelikethis', '0799_lovesexmagic', '0801_lovesong', '0802_lovestoned', '0803_lovestory', '0805_lovetheysay', '0806_mad', '0807_makemebetter', '0808_makethemoney', '0810_memoriesextended', '0811_miami2ibiza', '0812_miserybimbo', '0813_missindependent', '0814_mistake', '0815_mondaymondaymonday', '0817_morethanthis', '0819_myblood', '0820_mychickbad', '0823_mylifewouldsuckwithoutyou', '0825_myloveislikeastar', '0826_myohmy', '0827_needyounow', '0828_neoncathedral', '0829_nevertoolate', '0830_nightwatch', '0831_noair', '0832_noone', '0833_northshore', '0834_nosurprise', '0838_nowimallmessedup', '0842_obsessionstatic', '0844_ondirecting', '0846_onelove', '0848_onething', '0849_onlyyou', '0850_onlyyoucanlovemethisway', '0852_oursong', '0853_outonthetown', '0856_paperbackhead', '0863_pleasedontleavemedigi', '0865_pocketfulofsunshine', '0872_readyfortheweekend', '0874_realize', '0875_redbelt', '0878_replay', '0880_righthere', '0881_rightround', '0883_rocksatmywindow', '0884_rockstar', '0886_rudeboy', '0890_samemistakes', '0892_saveyoutonight', '0897_sentimentaltune', '0903_shakethatbubblenow', '0905_shocktoyoursystem', '0908_sober', '0910_solo', '0927_swing', '0930_takeyourtime', '0932_tattoo', '0936_tequieroacoustic', '0939_thecure', '0943_thetheayer', '0948_thislove', '0949_throwitinthebag', '0950_throwyourhandsup', '0951_thunder', '0953_tilltheworldends2', '0954_timeaftertimevan', '0955_tinylittlebows', '0958_turnmeup', '0961_unfaithfulfemme', '0962_usagainsttheworldjason', '0964_wakingupinvegas', '0970_whataboutnow', '0971_whatayawantfromme', '0973_whateverittakes', '0975_whathappensinvegas', '0981_whenyouregone', '0986_winner', '0987_withyou', '0988_withyoukovas', '0989_womanizer', '0991_worryaboutyounow', '0994_youbelongwithme', '0995_youfoundme', '0998_youregonnamissthis', '0999_yourheartisamuscle']
    # ['0001_12step', '0003_6foot7foot', '0004_abc', '0005_again', '0006_aint2proud2beg', '0008_america', '0009_americanmusic', '0010_andjusticeforall', '0011_areyouexperienced', '0012_aroundtheworld', '0014_babaoriley', '0015_babygotback', '0017_badromance', '0018_bassdownlow', '0020_becauseofyou', '0021_better', '0022_betteroffalone', '0024_billionaire', '0026_blackandyellow', '0027_blackened', '0028_blackmagic', '0032_boomboompow', '0035_boyfriend', '0036_breakingthegirl', '0037_breakyourheart', '0038_bringmetolife', '0039_bulletproof', '0040_bustamove', '0041_calabria', '0042_callme', '0043_callmemaybe', '0044_cameraeye', '0045_cantgetyou', '0046_castlesmadeofsand', '0047_chinacatsunflower', '0048_chingaling', '0049_closer', '0050_clubcanthandleme', '0053_commander', '0054_conceited', '0055_constantmotion', '0056_control', '0058_criticalacclaim', '0061_dabutt', '0062_dance', '0063_daysgoby', '0065_deadandbloated', '0066_decentdaysandnights', '0067_deep', '0068_devilsisland', '0069_digginmedown', '0070_dipitlow', '0071_dirtypool', '0072_discoinferno', '0073_disturbia', '0074_djgotusfallininlove', '0075_dontcha', '0077_dontsweat', '0078_donttouchme', '0079_dontyouwantme', '0080_down', '0081_downonme', '0082_dragosteadintei', '0083_dragthewaters', '0084_dropitlikeitshot', '0085_electricboogie', '0086_escapade', '0087_evacuate', '0088_everybody', '0091_fellonblackdays', '0092_fergalicious', '0093_fireburning', '0094_fireflies', '0095_firework', '0096_fivemagics', '0098_floods', '0099_forgetyou', '0100_futureperfecttense', '0102_gangstaluv', '0104_getbusy', '0105_getdownonit', '0106_getitshawty', '0107_getlow', '0109_geturfreakon', '0110_girlsandboys', '0111_girlsonfilm', '0112_giveitup', '0113_giveituptome', '0114_givemeeverything', '0115_gonnamakeyousweat', '0116_goodies', '0117_goodmorningblackfriday', '0118_grenade', '0119_gunpowderandlead', '0122_heardemall', '0123_heavyduty', '0124_hello', '0125_hellogoodmorning', '0126_heymami', '0127_hollabackgirl', '0128_homecoming', '0129_hotinherre', '0130_hotstuff', '0131_iamthebest', '0132_iceicebaby', '0133_ifeellove', '0135_igottafeeling', '0137_iknowyouwantme', '0138_ilikeit', '0140_impacto', '0141_indaclub', '0142_indestructible', '0143_informer', '0145_iwantyouback', '0149_johnnyguitar', '0150_justdance', '0151_kingofdancehall', '0152_kingofrock', '0153_lalaland', '0155_lapdance', '0156_lastnight', '0157_leanwitit', '0158_letitrock', '0159_letthemusicplay', '0160_likeag6', '0161_limelight', '0162_lobotomy', '0164_low', '0165_lucretia', '0167_makesomenoise', '0168_maneater', '0169_manicdepression', '0170_manyshadesofblack', '0172_marrythenight', '0173_massiveattack', '0174_maythisbelove', '0176_megasus', '0178_mountainman', '0179_moveslikejagger', '0181_mrsaxobeat', '0182_mycurse', '0183_mynameisjonas', '0184_myprerogative', '0186_nearlylostyou', '0189_neversaynever', '0190_newfang', '0191_newslang', '0195_nookie', '0197_nothinonyou', '0198_nowthatwefoundlove', '0199_numberofthebeast2', '0200_oceansize', '0201_oh', '0204_onlyamemory', '0208_opp', '0209_paparazzi', '0212_peoplegotalotofnerve', '0214_planetrock', '0215_pointofknowreturn', '0216_poison', '0217_pokerface', '0219_pondereplay', '0220_promiscuous', '0223_reallove', '0227_replay', '0229_ridingonthewind', '0231_rightthurr', '0233_rocknrollnightmare', '0234_roundandround', '0235_rudeboy', '0236_rumpshaker', '0237_run', '0239_sandm', '0240_sandstorm', '0241_satellite', '0242_satisfaction', '0243_saucyjack', '0244_sayaah', '0246_scenario', '0247_scream', '0248_screamingfor', '0249_sensualseduction', '0250_sexyandiknowit', '0251_sexychick', '0254_sixsixsix', '0255_sleepwalker', '0257_somebodytolove', '0258_sorry', '0259_spiceupyourlife', '0261_starships', '0264_stereolove', '0265_straightlines', '0266_straightup', '0267_superfreak', '0269_sweettalk', '0270_takeovercontrol', '0272_teachmehowtojerk', '0273_technologic', '0274_temperature', '0276_texasflood', '0278_thatsthahomie', '0279_thebreaks', '0280_thedayitriedtolive', '0281_theedgeofglory', '0282_thegreatsatan', '0283_thehumptydance', '0284_thehustle', '0285_thekill', '0286_theragelive', '0287_therewasatime', '0288_thewayiare', '0289_thisisexile', '0290_thisishowwedoit', '0291_thrasher', '0292_timeslikethese', '0293_tomsawyer2', '0294_tonighttonight', '0295_toxic', '0296_transmaniaconmc', '0297_troublecomesrunning', '0298_turnmeon', '0299_turnthebeataround', '0301_unclejohnsband', '0302_venus', '0303_wakeupdead', '0304_weaponofchoice', '0305_wenospeakamericano', '0306_werunthis', '0307_whatdoesntkillyou', '0308_whatislove', '0309_whatsmyname', '0310_whineup', '0312_whitefalconfuzz', '0314_whoomp', '0315_whygo', '0316_wildones', '0318_windup', '0321_wonderwall', '0322_wordup', '0323_yeah', '0324_yeah3x', '0325_ymca', '0326_yougotittherightstuff', '0327_youmakemefeel', '0328_youmakemefeeldc', '0329_youreajerk', '0330_50waystosaygoodbye', '0331_againstallodds', '0332_alejandro', '0333_allday', '0334_alltherightplaces', '0335_allyourlife', '0336_alonewithyou', '0337_alwayssomethingthere', '0338_animal', '0339_aslongasyouloveme', '0340_backintime', '0341_baggageclaim', '0342_bangpop', '0343_banjo', '0344_beautifullife', '0345_beinlovetonight', '0346_birthdaydress', '0347_bitch', '0348_blackout', '0349_blow', '0350_blowme', '0351_blowup', '0352_boogiewonderland', '0353_bottomsup', '0354_brandnewday', '0355_brokenhearted', '0356_californiagurls', '0357_cheers', '0359_cominghome', '0360_coolerthanme', '0361_countrymustbecountrywide', '0362_crazygirl', '0363_darkside', '0364_deuces', '0365_dirtroadanthem', '0366_domino', '0367_dontdreamitsover', '0368_dontwakemeup', '0370_downonme', '0371_downunder', '0372_drinkinmyhand', '0373_driveby', '0374_drunkonyou', '0375_dynamite', '0376_earthquake', '0377_eatdirt', '0379_et', '0380_everybodytalks', '0381_everythingyouwant', '0382_fatherofmine', '0383_feelsoclose', '0384_finally', '0385_fiveoclock', '0387_free', '0388_gangnamstyle', '0389_gettinoveryou', '0390_getyourselfbackhome', '0391_givemeallyourluvin', '0392_givemeyourhand', '0393_giveyourheartabreak', '0394_godgavemetoyou', '0395_gold', '0396_goodfeeling', '0397_goodlife', '0398_goodtime', '0399_gotmygirls', '0400_heatofthemoment', '0401_hello', '0402_herestous', '0404_holditagainstme', '0405_home', '0406_honeybee', '0407_humpinaround', '0408_icanonlyimagine', '0409_idontlikeyou', '0410_idontwantthisnighttoend', '0411_ifihadyou', '0412_ifitmakesyouhappy', '0413_ifyoucouldonlysee', '0414_illbe', '0416_impossible', '0417_imtoosexy', '0418_inthedark', '0419_itgirl', '0420_itsrainingmen', '0421_ittakestwo', '0422_itwillrain', '0423_iwannago', '0424_justadream', '0425_justakiss', '0426_justcantgetenough', '0427_justthewayyouare', '0428_kingofanything', '0429_kissinu', '0430_lastfridaynight', '0431_letitrain', '0432_letsgo', '0433_lighters', '0434_lights', '0435_lookatmenow', '0436_loveactually', '0437_lovelikewoe', '0438_loverlover', '0439_lovethewayyoulie', '0440_loveyoulikealovesong', '0442_lullaby', '0443_magic', '0444_maybe', '0445_memories', '0446_midnightcity', '0447_mindyourmanners', '0448_mine', '0449_misery', '0450_more', '0451_mrknowitall', '0452_myfirstkiss', '0453_mygirl', '0454_myohmy', '0455_nevertearusapart', '0456_nohands', '0457_nothing', '0458_notorious', '0459_notoveryou', '0460_numb', '0461_obsession', '0462_oldalabama', '0463_onemorenight', '0465_onething', '0466_onthedarkside', '0467_onthefloor', '0468_openarms', '0470_ours', '0471_paradise', '0472_partofme', '0473_payphone', '0474_pleasedontgo', '0475_poisonedwithlove', '0476_pontoon', '0477_poundthealarm', '0478_prayforyou', '0479_pricetag', '0480_raiseyourglass', '0481_reallove', '0482_redsolocup', '0483_ridinsolo', '0484_rockthatbody', '0485_rollinginthedeep', '0487_secrets', '0488_september', '0489_shakeyourbody', '0490_showme', '0491_skyscraper', '0492_sm', '0494_somebodythatiusedtoknow', '0495_someonelikeyou', '0496_sparksfly', '0497_speakers', '0498_springsteen', '0499_starbuckssmile', '0500_stay', '0501_stayawhile', '0502_stronger', '0503_stucklikeglue', '0504_suspiciousminds', '0505_takeitoff', '0506_takemehometonight', '0507_teenagedream', '0508_tennesseeme', '0509_thelazysong', '0510_theonethatgotaway', '0511_theonlyexception', '0512_theshowgoeson', '0513_thetime', '0514_thetimeofmylife', '0515_thisafternoon', '0517_tilltheworldends', '0518_timeaftertime', '0519_titanium', '0520_tonight', '0521_tooclose', '0522_turnupthemusic', '0523_ucanttouchthis', '0524_undoit', '0525_venus', '0526_waitingoutsidethelines', '0527_wantuback', '0528_werunthenight', '0529_werwhower', '0530_whatthehell', '0531_whenwestandtogether', '0532_whistle', '0533_whosays', '0534_wideawake', '0535_withoutyou', '0536_workout', '0537_writteninthestars', '0538_you', '0539_youandi', '0540_youandme', '0541_youdroppedabombonme', '0542_yougetwhatyougive', '0543_yougonnafly', '0544_yougottabe', '0545_youkeepmehangingon', '0546_18days', '0548_2getherextended', '0550_aboutyounow', '0551_accordingtoyou', '0552_addicted', '0554_allalright', '0556_allaroundtheworld', '0558_allnightlong', '0559_alltherightmoves', '0560_almostlovenow', '0561_alorsondanse', '0562_alreadygone', '0564_americanboybenny', '0565_americanhoney', '0566_americanwoman', '0567_anythingcouldhappen', '0568_apologize', '0569_arrow', '0570_aslongasyouloveme', '0571_atlantis', '0572_awake', '0573_babyludacris', '0574_badromanceappboyz', '0575_badromanceskrillexremix', '0576_barbrastreisand', '0577_battlefieldmainversion', '0578_bealright', '0579_beatitstudioversion', '0580_beautiful', '0582_beautifulday', '0583_beautifulgirls', '0585_beautyandabeat', '0586_bed', '0587_bedrock', '0588_believe', '0589_bestieverhad', '0590_betterintime', '0591_biggirlsdontcry', '0592_blameit', '0594_bleedinglove', '0595_blonde', '0597_boomboomguetta', '0598_boomboompow', '0599_boyfriend', '0600_breakeven', '0602_breaktheice', '0603_breaktheicejasonnevinsmix', '0604_bustitbabypart2', '0607_cantholdus', '0608_canttellmenothing', '0609_carryon', '0610_castle', '0611_catchingfeelings', '0614_cinema', '0615_circus', '0616_citygirl', '0617_citylights', '0618_clocks', '0619_closer', '0620_closertands', '0621_clubcanthandleme', '0622_clumsy', '0623_comeongethigher', '0625_complicated', '0626_coolerthanme', '0627_cowboyboots', '0628_crankthatsouljaboy', '0629_crush', '0630_crushcrushcrush', '0631_curiosity', '0632_cyclone', '0633_dancingmachine', '0634_dangerous', '0635_daynnite', '0636_daynnitecrookersremix', '0637_daynniteremix', '0638_deadandgone', '0639_dieinyourarms', '0640_disturbia2', '0642_dontconfess', '0643_dontrush', '0644_dontsayaword', '0645_dontstopthemusic', '0647_dontthinkidontthinkaboutit', '0648_donttrustme', '0649_down', '0650_downlilwayne', '0651_doyouremember', '0652_drovemewild', '0653_dynamite', '0654_dynamitemixinmarc', '0655_earthquakey', '0657_enursbonfire', '0659_everybody', '0661_everythingaboutyou', '0662_everytimewetouch', '0663_explosions', '0664_eyesextendedmix', '0665_fall', '0666_fallin', '0668_fallinganthem', '0669_fastforwardffwd', '0670_fearless', '0671_feedback', '0672_feelitinmybones', '0673_feelsliketonight', '0676_figure8', '0677_fireburning', '0678_fireflies', '0679_firsttime', '0680_fixaheart', '0681_flashinglights', '0682_forever', '0683_foreverextendedmix', '0685_forgottolaugh', '0686_fortheloveofadaughter', '0687_getouttamyway', '0688_getthepartystarted', '0689_gettinoveryou', '0690_givemeeverything', '0691_givesyouhell', '0692_giveyourheartabreakdos', '0693_gold', '0695_goodbyegoodbye', '0696_goodgirlsgobad', '0697_goodlife', '0698_goodtime', '0699_gotmoney', '0700_gottabesomebody', '0701_gottabeyou', '0702_greenlight', '0704_guitarstringweddingring', '0705_haditall', '0706_halcyon', '0707_halfwaygone', '0708_haloradioedit', '0709_hangingon', '0710_hardjeezy', '0712_heartless', '0713_heartofgoldnow', '0714_hell', '0715_hellodadalife', '0716_heroheroine', '0717_heysoulsister', '0718_heytheredelilah', '0719_hipsdontlie', '0721_holdup', '0723_hotelroomservice', '0724_hotmessdj', '0725_hotncold', '0726_hotplay', '0727_howcomeyoudontwantme', '0728_howyouremindme', '0729_hurricane', '0730_hurtssogood', '0731_icouldntbeyourfriend', '0732_idontwannabeinlove', '0733_ifihadyou', '0734_iftodaywasyourlastday', '0735_ifuseeamy', '0736_igotafeeling', '0737_ihatethispart', '0738_ihearnoises', '0739_ikissedagirl', '0740_ikissedagirljasonnevins', '0741_iknowyoucare', '0742_iknowyouwantme2', '0743_iminloveiwannadoit', '0744_immabe', '0745_immabewolf', '0746_imnotyourhero', '0748_imyours', '0749_indestructible2', '0750_ineedyourlove', '0751_inlovewithagirl', '0752_inmyhead', '0753_inreallife', '0756_istillmissyou', '0758_itgetsbetter', '0759_itgirljasonnevins', '0760_iwannarock', '0761_iwant', '0762_iwantitthatway', '0763_iwasafool', '0764_iwish', '0765_jimmyiovine', '0766_justdance', '0767_justfine', '0768_keepsgettinbetter', '0769_killa', '0770_kingofmycastle', '0771_kisskissmainversion', '0773_kissmethruthephone', '0774_knockyoudown', '0775_knockyoudownbimbojones', '0776_krazy', '0777_lebump', '0778_letitrock', '0781_lettinggoduttylove', '0782_lifeafteryou', '0783_lighton', '0784_lightweight', '0785_littleliesnow', '0786_livelikeweredying', '0787_liveyourlife', '0789_livingroom', '0790_locapeople', '0791_lollipop', '0792_lovedrunk', '0793_lovegame', '0794_lovegamedave', '0798_lovelikethis', '0799_lovesexmagic', '0801_lovesong', '0802_lovestoned', '0803_lovestory', '0805_lovetheysay', '0806_mad', '0807_makemebetter', '0808_makethemoney', '0810_memoriesextended', '0811_miami2ibiza', '0812_miserybimbo', '0813_missindependent', '0814_mistake', '0815_mondaymondaymonday', '0817_morethanthis', '0819_myblood', '0820_mychickbad', '0823_mylifewouldsuckwithoutyou', '0825_myloveislikeastar', '0826_myohmy', '0827_needyounow', '0828_neoncathedral', '0829_nevertoolate', '0830_nightwatch', '0831_noair', '0832_noone', '0833_northshore', '0834_nosurprise', '0838_nowimallmessedup', '0842_obsessionstatic', '0844_ondirecting', '0846_onelove', '0848_onething', '0849_onlyyou', '0850_onlyyoucanlovemethisway', '0852_oursong', '0853_outonthetown', '0856_paperbackhead', '0863_pleasedontleavemedigi', '0865_pocketfulofsunshine', '0872_readyfortheweekend', '0874_realize', '0875_redbelt', '0878_replay', '0880_righthere', '0881_rightround', '0883_rocksatmywindow', '0884_rockstar', '0886_rudeboy', '0890_samemistakes', '0892_saveyoutonight', '0897_sentimentaltune', '0903_shakethatbubblenow', '0905_shocktoyoursystem', '0908_sober', '0910_solo', '0927_swing', '0930_takeyourtime', '0932_tattoo', '0936_tequieroacoustic', '0939_thecure', '0943_thetheayer', '0948_thislove', '0949_throwitinthebag', '0950_throwyourhandsup', '0951_thunder', '0953_tilltheworldends2', '0954_timeaftertimevan', '0955_tinylittlebows', '0958_turnmeup', '0961_unfaithfulfemme', '0962_usagainsttheworldjason', '0964_wakingupinvegas', '0970_whataboutnow', '0971_whatayawantfromme', '0973_whateverittakes', '0975_whathappensinvegas', '0981_whenyouregone', '0986_winner', '0987_withyou', '0988_withyoukovas', '0989_womanizer', '0991_worryaboutyounow', '0994_youbelongwithme', '0995_youfoundme', '0998_youregonnamissthis', '0999_yourheartisamuscle']
   
    dataset_path = run_salami_preprocessing(
        target_song_ids=TARGET_SONG_IDS,
        max_len=935
    )
    
    print(f"\n✅ Final dataset ready: {dataset_path}")
    print("🎯 Expected performance improvement: F1_seg 0.428 → 0.50-0.57")