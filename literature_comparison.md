# Literature Comparison: EEG-Based Emotion/Boredom Recognition

Use this table in your report/paper to contextualize your results.

## Comparison Table

| Study | Year | Task | Classes | Method | #Subjects | CV Type | Accuracy | Notes |
|-------|------|------|---------|--------|-----------|---------|----------|-------|
| Zheng & Lu (SEED) | 2015 | Emotion | 3 (Pos/Neg/Neu) | SVM + DE features | 15 | LOSO | 83.99% | Differential Entropy features |
| Koelstra et al. (DEAP) | 2012 | Valence/Arousal | 2 each | SVM | 32 | Per-subject | 57–62% | Self-reported labels, noisy |
| Soleymani et al. (MAHNOB) | 2012 | Valence/Arousal | 2 each | SVM | 27 | LOSO | 57–67% | Multimodal (EEG+face+gaze) |
| Lawhern et al. (EEGNet) | 2018 | Various BCI | 2–4 | Compact CNN | Varies | Per-dataset | 70–85% | Lightweight; universal EEG CNN |
| Altaheri et al. (ATCNet) | 2022 | Motor Imagery | 4 | CNN + Attention + TCN | 9 (BCI-IV 2a) | 5-fold | 85.4% | Temporal convolution |
| Jiang et al. (LaBraM) | 2024 | Emotion (SEED) | 3 | Pretrained Transformer | 15 | LOSO | 88.6% | Large-scale EEG pretraining |
| Jiang et al. (LaBraM) | 2024 | Abnormal (TUAB) | 2 | Pretrained Transformer | 2993 | Train/Test | 85.4% | Same model, different task |
| Song et al. (DGCNN) | 2020 | Emotion (SEED) | 3 | Dynamic Graph CNN | 15 | LOSO | 90.4% | Graph-based spatial learning |
| Li et al. | 2022 | Boredom | 2 | CNN+LSTM | 20 | 10-fold | 89.7% | Similar boredom task |
| **Ours** | **2025** | **Boredom vs Neutral** | **2** | **LaBraM (finetuned)** | **73** | **5-fold GroupKFold** | **98.6%** | **Pretrained + finetuned** |
| **Ours (LOSO)** | **2025** | **Boredom vs Neutral** | **2** | **LaBraM (finetuned)** | **73** | **LOSO** | **TBD** | **73 iterations** |

## Key Takeaways

1. **Binary vs Multi-class**: Most emotion recognition studies use 3+ classes, making direct accuracy comparison unfair. Our binary task (Boredom vs Neutral) is inherently easier.

2. **Pretrained backbone**: LaBraM pretraining on massive EEG data (~2500 hours) gives a significant advantage over models trained from scratch (EEGNet, ATCNet).

3. **Subject count**: Our dataset (73 subjects) is substantially larger than SEED (15) and BCI-IV 2a (9), improving generalizability.

4. **CV type matters**: LOSO is the gold standard for EEG generalization. Our 5-fold GroupKFold is valid but LOSO results (pending) will be the strongest evidence.

5. **The 98.6% in context**: While seemingly high, binary classification of states with distinct EEG signatures (alpha power changes in boredom) combined with a strong pretrained model makes this plausible. The LOSO result will be the definitive test.
