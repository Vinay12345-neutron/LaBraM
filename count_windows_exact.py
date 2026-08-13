import json
import h5py

for i in range(5):
    with open(f"boredom_split_fold{i}.json") as f:
        d = json.load(f)
    
    counts = {"train": 0, "val": 0, "test": 0}
    b_subjects = {"train": set(), "val": set(), "test": set()}
    n_subjects = {"train": set(), "val": set(), "test": set()}

    for split in ["train", "val", "test"]:
        for item in d[split]:
            filepath = item["file"]
            label = item["label"] # 1=Boredom, 0=Neutral
            
            # Extract subject ID
            filename = filepath.split("/")[-1]
            subject_id = filename.split("_")[0]
            
            if label == 1:
                b_subjects[split].add(subject_id)
            else:
                n_subjects[split].add(subject_id)

            try:
                with h5py.File(filepath, "r") as f:
                    group_key = list(f.keys())[0]
                    subject_len = f[group_key]["eeg"].shape[1]
                    num_windows = (subject_len - 512) // 512 + 1
                    counts[split] += num_windows
            except Exception as e:
                pass
                
    # Formatting for markdown table
    print(f"| Fold {i} | {len(b_subjects['train'])}B, {len(n_subjects['train'])}N | {len(b_subjects['val'])}B, {len(n_subjects['val'])}N | {len(b_subjects['test'])}B, {len(n_subjects['test'])}N | {counts['train']} | {counts['val']} | {counts['test']} |")
