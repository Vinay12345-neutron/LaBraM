import json

for i in range(5):
    try:
        with open(f"runs/boredom_cv/fold{i}/test_predictions.json") as f:
            lines = f.read().strip().split("\n")
            preds = json.loads(lines[-1]) # get the last epoch
            
        # Wait, test_predictions.json contains arrays: "preds", "trues"
        # but it doesn't contain the filenames!
        # Oh, it doesn't have the subject IDs attached.
        print(f"Fold {i} length:", len(preds["trues"]))
    except Exception as e:
        pass
