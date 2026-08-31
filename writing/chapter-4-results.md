
# Chapter 4: Results

## 4.1 Introduction

This chapter presents the results produced by applying the methodology described in Chapter 3. Results are reported in the same order the methodology introduced them: classification performance for all three architectures on both tasks, the comparison of the three DDI incorporation strategies, the sample-size correction observed in the Grad-CAM faithfulness scores, and the comparison between Grad-CAM and SHAP faithfulness at increasing sample sizes, concluding with the paired confidence interval analysis that settles which architecture favours which explainability method. Interpretation of these results against the claims and gaps identified in the literature review is left to Chapter 5. This chapter reports what was found.

## 4.2 Classification Performance

### 4.2.1 Seven-Class Classification Results

Table 4.1 reports each architecture's performance on the primary seven-class task, on both the internal HAM10000 validation split and the independent ISIC2018 test set.

**Table 4.1: Seven-class classification performance**

| Architecture | Validation accuracy | Validation kappa | Validation ROC-AUC | ISIC2018 test accuracy |
|---|---|---|---|---|
| ResNet-50 | 65.8% | 0.470 | 0.935 | 65.7% |
| EfficientNetB4 | 63.0% | 0.431 | 0.913 | 61.4% |
| VGG-16 | 64.0% | 0.447 | 0.931 | 66.3% |

ResNet-50 achieved the highest validation accuracy and kappa among the three architectures, though the margin over VGG-16 is small (1.8 percentage points on accuracy, 0.023 on kappa). VGG-16 achieved the highest accuracy on the independent ISIC2018 test set, ahead of ResNet-50 by 0.6 percentage points and EfficientNetB4 by 4.9 percentage points. EfficientNetB4 recorded the lowest score on every metric in this table.

For all three architectures, ISIC2018 test accuracy tracked internal validation accuracy closely, within 1.6 percentage points in every case. This close agreement between an internal split and a fully independent, lesion-disjoint test set indicates that none of the three models had been overfit to the internal validation split. Figure 4.1 presents the same comparison graphically.

![Figure 4.1: Seven-class accuracy by architecture, HAM10000 validation versus ISIC2018 test](figures/fig_4_1_seven_class_accuracy.png)

### 4.2.2 Binary Classification Results

Table 4.2 reports each architecture's performance on the binary malignant or benign task, using the joint training strategy described in Chapter 3, which the results below establish as the recommended binary-task model.

**Table 4.2: Binary classification performance, joint training strategy**

| Architecture | Accuracy | Kappa | ROC-AUC |
|---|---|---|---|
| ResNet-50 | 81.7% | 0.502 | 0.879 |
| EfficientNetB4 | 74.3% | 0.410 | 0.873 |
| VGG-16 | 81.5% | 0.492 | 0.881 |

ResNet-50 and VGG-16 performed almost identically on this task, separated by only 0.2 percentage points of accuracy and 0.010 of kappa. EfficientNetB4 trailed both by a wider margin than it did on the seven-class task, 7.2 to 7.4 percentage points of accuracy behind the other two. Figure 4.2 shows accuracy and kappa side by side for all three architectures.

![Figure 4.2: Binary classification accuracy and Cohen's kappa by architecture, joint training strategy](figures/fig_4_2_binary_accuracy_kappa.png)

## 4.3 DDI Fairness Evaluation Results

### 4.3.1 Zero-Shot Generalisation Results

Table 4.3 reports each HAM10000-only binary model's performance on the full 656-image DDI set, without any DDI training.

**Table 4.3: Zero-shot generalisation to DDI**

| Architecture | DDI accuracy | DDI kappa | DDI ROC-AUC | HAM10000 malignant recall | DDI malignant recall |
|---|---|---|---|---|---|
| ResNet-50 | 75.3% | 0.159 | 0.654 | 0.799 | 0.158 |
| EfficientNetB4 | 62.5% | 0.138 | 0.592 | 0.871 | 0.480 |
| VGG-16 | 73.2% | 0.090 | 0.598 | 0.799 | 0.123 |

Every architecture's malignant recall collapsed when moving from HAM10000 to DDI without any adaptation, though the size of the collapse varied considerably. VGG-16 showed the largest relative drop, from 0.799 to 0.123. EfficientNetB4 showed both the highest starting recall on HAM10000 (0.871) and the highest recall retained on DDI (0.480), making it the best generaliser of the three architectures on this specific measure despite recording the lowest overall DDI accuracy and kappa.

### 4.3.2 Sequential Fine-Tuning Results

Table 4.4 reports each architecture's performance after fine-tuning its HAM10000-only baseline on DDI's own training split, evaluated on DDI's held-out validation split, together with how much HAM10000 validation accuracy each model retained.

**Table 4.4: Sequential fine-tuning on DDI**

| Architecture | DDI held-out accuracy | DDI held-out kappa | DDI held-out ROC-AUC | HAM10000 retention |
|---|---|---|---|---|
| ResNet-50 | 71.7% | 0.167 | 0.584 | 75.4% |
| EfficientNetB4 | 66.7% | 0.083 | 0.615 | 77.5% |
| VGG-16 | 70.7% | 0.071 | 0.671 | 80.0% |

Fine-tuning did not produce a uniform outcome across architectures. Compared against the zero-shot malignant recall figures in Table 4.3, ResNet-50 and VGG-16 both saw their malignant recall on DDI rise after fine-tuning, while EfficientNetB4's malignant recall fell after fine-tuning despite this being the architecture that generalised best zero-shot. EfficientNetB4's kappa on the held-out split, 0.083, is also the lowest of the three architectures under this strategy.

### 4.3.3 Joint Training Results

Table 4.5 reports each architecture's performance when trained on HAM10000 and DDI together from the outset, evaluated on the identical DDI held-out split used in Table 4.4.

**Table 4.5: Joint training on HAM10000 and DDI**

| Architecture | DDI held-out accuracy | DDI held-out kappa | DDI held-out ROC-AUC |
|---|---|---|---|
| ResNet-50 | 74.7% | 0.356 | 0.764 |
| EfficientNetB4 | 65.7% | 0.211 | 0.698 |
| VGG-16 | 70.7% | 0.304 | 0.733 |

### 4.3.4 Comparative Summary Across Strategies

Table 4.6 places the three strategies side by side using their kappa scores on the DDI held-out split, since kappa was established in Chapter 3 as the decisive metric for this comparison.

**Table 4.6: Kappa on the DDI held-out split, by strategy and architecture**

| Architecture | Fine-tuned kappa | Joint kappa | Difference |
|---|---|---|---|
| ResNet-50 | 0.167 | 0.356 | +0.189 |
| EfficientNetB4 | 0.083 | 0.211 | +0.128 |
| VGG-16 | 0.071 | 0.304 | +0.233 |

Joint training produced a higher kappa than sequential fine-tuning for every architecture, with the joint model's kappa between 1.5 and 4.3 times the fine-tuned model's kappa depending on architecture. The same ordering holds on ROC-AUC: joint training led fine-tuning by 0.180 for ResNet-50, 0.083 for EfficientNetB4, and 0.062 for VGG-16. Figure 4.3 shows this comparison for all three architectures.

![Figure 4.3: Cohen's kappa on the DDI held-out split, fine-tuned versus joint training](figures/fig_4_3_ddi_kappa_strategy_comparison.png)

### 4.3.5 Skin-Tone-Stratified Results

Table 4.7 breaks the joint model's DDI held-out accuracy down by Fitzpatrick skin-tone band, using the strategy recommended above.

**Table 4.7: Joint model accuracy by Fitzpatrick skin-tone band, DDI held-out split**

| Architecture | FST I to II | FST III to IV | FST V to VI |
|---|---|---|---|
| ResNet-50 | 68.8% | 72.2% | 83.9% |
| EfficientNetB4 | 56.3% | 75.0% | 64.5% |
| VGG-16 | 71.9% | 63.9% | 77.4% |

Each Fitzpatrick band in this table contains between 31 and 36 images, so these figures should be read as directional rather than precise. No architecture shows a monotonic decline in accuracy from lighter to darker skin tones. ResNet-50's highest accuracy band is FST V to VI, the darkest band, at 83.9%. EfficientNetB4's lowest accuracy band is FST I to II, the lightest band, at 56.3%. VGG-16's lowest band is FST III to IV, the middle band. Figure 4.4 presents these results graphically.

![Figure 4.4: Joint model accuracy by Fitzpatrick skin-tone band and architecture, DDI held-out split](figures/fig_4_4_skin_tone_stratified_accuracy.png)

## 4.4 Grad-CAM Faithfulness Results

### 4.4.1 Initial Sample and Correction at Larger Sample Size

Table 4.8 compares each model's Grad-CAM faithfulness score at the original 30-image sample against the same score recomputed at 150 images.

**Table 4.8: Grad-CAM mean IoU, 30-image sample versus 150-image sample**

| Model | n=30 IoU | n=150 IoU | Change |
|---|---|---|---|
| ResNet-50, seven-class | 0.253 | 0.294 | +16.5% |
| ResNet-50, binary | 0.215 | 0.157 | −27.0% |
| EfficientNetB4, seven-class | 0.239 | 0.259 | +8.4% |
| EfficientNetB4, binary | 0.167 | 0.141 | −15.6% |
| VGG-16, seven-class | 0.196 | 0.196 | unchanged |
| VGG-16, binary | 0.154 | 0.115 | −25.3% |

Every binary-task model's faithfulness score fell by between 15.6% and 27.0% when the sample size increased from 30 to 150 images. Every seven-class model's score either rose or stayed the same over the same increase. This pattern, a consistent direction of change within each task rather than a mix of increases and decreases across all six models, is consistent with the binary task's smaller pool of validation images making a 30-image sample less representative of that task specifically, rather than with a general unreliability affecting all six models equally. Figure 4.5 presents both tasks side by side.

![Figure 4.5: Grad-CAM mean IoU at n=30 versus n=150, seven-class and binary tasks](figures/fig_4_5_gradcam_sample_size_correction.png)

## 4.5 SHAP versus Grad-CAM Faithfulness Results

### 4.5.1 Comparison Across Sample Sizes

Table 4.9 reports the mean IoU achieved by Grad-CAM and by SHAP for each architecture, at each sample size tested, culminating in the largest sample size that could be run for each architecture given the memory constraint described in Chapter 3.

**Table 4.9: Grad-CAM and SHAP mean IoU across sample sizes**

| Architecture | n | Grad-CAM IoU | SHAP IoU |
|---|---|---|---|
| ResNet-50 | 15 | 0.293 | 0.259 |
| ResNet-50 | 150 | 0.294 | 0.215 |
| ResNet-50 | 500 | 0.305 | 0.208 |
| EfficientNetB4 | 15 | 0.185 | 0.273 |
| EfficientNetB4 | 150 | 0.259 | 0.255 |
| EfficientNetB4 | 400 | 0.255 | 0.241 |
| VGG-16 | 15 | 0.200 | 0.218 |
| VGG-16 | 150 | 0.195 | 0.255 |
| VGG-16 | 500 | 0.202 | 0.255 |

ResNet-50's Grad-CAM score exceeded its SHAP score at every sample size tested, and the size of that gap grew slightly as the sample size increased, from 0.034 at 15 images to 0.097 at 500 images. VGG-16's SHAP score exceeded its Grad-CAM score at every sample size tested, growing from a narrow 0.018 at 15 images to 0.053 at 500 images. EfficientNetB4 is the only architecture whose ranking changed with sample size: Grad-CAM trailed SHAP by 0.088 at 15 images, then led SHAP by 0.004 at 150 images and by 0.014 at 400 images. Figure 4.6 traces each architecture's two scores across every sample size tested.

![Figure 4.6: Grad-CAM and SHAP mean IoU across sample sizes, by architecture](figures/fig_4_6_gradcam_shap_across_sample_sizes.png)

### 4.5.2 Confidence Interval Analysis

To determine whether the differences in Table 4.9 reflect a genuine effect rather than sampling noise, the per-image difference between each model's Grad-CAM and SHAP IoU was computed at the largest sample size run for each architecture, and used to construct a paired 95% confidence interval on the mean difference. Table 4.10 reports the result.

**Table 4.10: Paired 95% confidence interval on the Grad-CAM minus SHAP IoU gap**

| Architecture | n | Mean gap | 95% confidence interval | Excludes zero |
|---|---|---|---|---|
| ResNet-50 | 500 | +0.097 | 0.078 to 0.115 | Yes |
| VGG-16 | 500 | −0.053 | −0.067 to −0.038 | Yes |
| EfficientNetB4 | 400 | +0.014 | −0.003 to 0.032 | No |
| EfficientNetB4 | 200 | +0.007 | −0.018 to 0.032 | No |

ResNet-50's and VGG-16's confidence intervals both exclude zero, in opposite directions, indicating a statistically supported difference between the two explainability methods for each of these two architectures. EfficientNetB4's confidence interval includes zero at both sample sizes tested, 400 and 200 images, indicating that the two methods cannot be distinguished for this architecture at either sample size. Since this result was obtained independently at two different sample sizes with the same outcome, it is treated as a stable finding for this architecture rather than as a result that a still larger sample might yet resolve one way or the other. Figure 4.7 presents all four intervals on a common scale, with the dashed line marking zero.

![Figure 4.7: Paired 95% confidence interval on the Grad-CAM minus SHAP mean IoU gap](figures/fig_4_7_paired_confidence_intervals.png)

## 4.6 Summary of Findings

ResNet-50 achieved the strongest seven-class validation performance and joint binary performance among the three architectures, though its lead over VGG-16 was narrow on most metrics, and VGG-16 achieved the strongest ISIC2018 test accuracy. EfficientNetB4 trailed both architectures on nearly every classification metric, on both tasks.

Across all three DDI incorporation strategies, joint training produced the highest kappa on the DDI held-out split for every architecture, by a substantial margin over sequential fine-tuning in every case. Zero-shot evaluation on the full DDI set showed a severe collapse in malignant recall for every architecture, of differing sizes, with EfficientNetB4 retaining the most malignant recall despite being the weakest architecture by several other measures.

Grad-CAM faithfulness scores computed on a small sample were shown not to be reliable on their own: every binary-task model's score changed substantially, always downward, when the sample size was increased from 30 to 150 images. This same sensitivity was found again in the SHAP comparison, where an apparent SHAP advantage for EfficientNetB4 at 15 images did not survive scaling to 150, 200, or 400 images, while a smaller apparent SHAP advantage for VGG-16 at 15 images held and grew at every larger sample size tested. The confidence interval analysis in Table 4.10 supports two directional findings, ResNet-50 favouring Grad-CAM and VGG-16 favouring SHAP, and one genuine tie, EfficientNetB4 showing no reliable difference between the two methods.
