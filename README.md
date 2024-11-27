<div align="center">

# SoccerNet Team Ball Action Spotting

</div>

Welcome to the SoccerNet Development Kit for the Team Ball Action Spotting challenge. This challenge is an evolution of the original Action Spotting and Ball Action Spotting tasks, which are described in more detail [here](https://github.com/SoccerNet/sn-spotting?tab=readme-ov-file).

<p align="center"><img src="figures/task.png"></p>

⚽ **What is Action Spotting?**  
Action spotting involves the identification and precise localization of actions within an untrimmed video. Given a video input, the objective is to detect and accurately locate all the actions occurring throughout the video.

- The original **SoccerNet Action Spotting** task focused on identifying 17 sparse actions across soccer broadcasts, spanning a total of 550 football games. Key features included sparse annotations and a looser evaluation metric.
- The **SoccerNet Ball Action Spotting** task introduced a new set of soccer videos, with denser annotations for 12 distinct actions. A stricter metric was also employed, with tolerances of just up to 1 second.
- The **SoccerNet Team Ball Action Spotting** task is a direct extension of the Ball Action Spotting task. It uses the same games and actions but adds the challenge of identifying which team (left or right in the video) performed the action.

In this repository, we focus on the latest specification of the task, using an adaptation of the SoccerNet 2024 Ball Action Spotting challenge winner model, [T-DEED](https://arxiv.org/abs/2404.05392), as our baseline. The evaluation metric is also adapted to assess whether methods correctly predict the team performing the action. Further details on this adaptation are provided below.

## T-DEED - Temporal-Discriminability Enhancer Encoder-Decoder

[![arXiv](https://img.shields.io/badge/arXiv-2404.05392-red)](https://arxiv.org/abs/2404.05392)

As mentioned earlier, we use the [T-DEED](https://arxiv.org/abs/2404.05392) model as the baseline for our task. T-DEED is a Temporal-Discriminability Enhancer Encoder-Decoder designed for end-to-end training with the goal of improving token discriminability. Further details about the model can be found in the original paper and in its dedicated [repository](https://github.com/arturxe2/T-DEED). In this repository, we provide the adaptation of T-DEED for the Team Ball Action Spotting task.

<p align="center"><img src="figures/modelArchitecture_v2.png"></p>

To adapt T-DEED for predicting the team performing the action, we add an additional prediction head with a sigmoid activation function, and incorporate a team loss during training based on Binary Cross-Entropy loss. T-DEED’s evaluation in the SoccerNet 2024 Ball Action Spotting challenge benefited from joint training on both the SoccerNet Action Spotting and SoccerNet Ball Action Spotting datasets. We follow the same approach here to leverage the larger original dataset, which helps in learning the actions for Ball Action Spotting.

### Steps for Using the Repository:

1. **Download the datasets and labels**:
   Follow the instructions provided in this [link](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025/tree/main) to download the SoccerNet Action Spotting and SoccerNet Ball Action Spotting datasets, including videos and labels. While the labels and SoccerNet Ball Action Spotting videos can be downloaded directly from the link, the instructions for downloading SoccerNet Action Spotting videos can be found on this [webpage](https://www.soccer-net.org/data). If you have labels from earlier versions of the challenge, ensure that you update them, as the new versions include team annotations for each action.

2. **Update label paths**:  
   Modify the `labels_path.txt` files in [`/data/soccernet`](/data/soccernet/) and [`/data/soccernetball`](/data/soccernetball/) to point to the folders containing the labels for each dataset.

3. **Extract frames**:  
   Extract frames for both SoccerNet Action Spotting and SoccerNet Ball Action Spotting using the [`extract_frames_sn.py`](/extract_frames_sn.py) and [`extract_frames_snb.py`](/extract_frames_snb.py) scripts.

4. **Update the config file**:  
   Update the [config file](/config/SoccerNetBall/SoccerNetBall_baseline.json) to set the paths for frame directories (for both datasets), and checkpoint and predictions saving locations. For details on the configuration parameters, refer to the corresponding [README](/config/SoccerNetBall/README.md).

5. **Run initial training setup**:  
   Execute the [`train_tdeed_bas.py`](/train_tdeed_bas.py) script, specifying the model name corresponding to the configuration file (`SoccerNetBall_baseline` in this case). For the first run, set the `store_mode` parameter in the configuration to "store". This will partition the untrimmed videos into clips and store information such as the starting frame and processed labels, enabling faster data loading during training.

6. **Train the model**:  
   After the initial setup, run `train_tdeed_bas.py` again with `store_mode` set to "load" to train the model. The script will train on the `train` splits, use the `validation` splits for early stopping, evaluate on the `test` split, and generate predictions for the `challenge` split.

## Evaluation

To evaluate this task, we use a modified version of the mAP (mean Average Precision) metric that has been employed in previous Action Spotting challenges. In this adaptation, we compute the AP (Average Precision) for each combination of action (12 different actions) and team (left or right). The final AP for each action is calculated as the weighted average between both team sides, with the weights determined by the number of ground-truth observations. Lastly, the overall Team mAP metric is obtained by averaging the AP values across all 12 actions. As in the previous Ball Action Spotting challenge, a tolerance of 1 second is applied.

You can check [/util/eval.py](/util/eval.py) and [/util/score.py](/util/score.py) for the implementation of the evaluation metric. You can evaluate your models on the test split using this repository, or submit your predictions for both the test and challenge splits to the relevant competitions ([test](https://www.codabench.org/competitions/4418/) and [challenge](https://www.codabench.org/competitions/4417/)) on Codabench. Additional details regarding the structure of the submission file can be found on the respective Codabench competition pages.


## 2025 Team Ball Action Spotting Challenge

|      |Test split|      |Challenge split|      |
|------|----------|------|---------------|------|
|Method|Team mAP@1|mAP@1 |Team mAP@1     |mAP@1 |
|[T-DEED](https://arxiv.org/abs/2404.05392) (baseline)|47.18|53.59|51.72|58.38|

Here, we present the results of the provided T-DEED baseline. Challenge yourself by pushing the limits of your methods and participating in our upcoming [2025 Challenges](https://www.soccer-net.org/challenges/2025). The official rules and submission guidelines can be found in [ChallengeRules.md](/ChallengeRules.md). You can submit your predictions to the Codabench evaluation server for both the [test](https://www.codabench.org/competitions/4418/) and [challenge](https://www.codabench.org/competitions/4417/) competitions.

A checkpoint of the baseline is available at the following [link](https://drive.google.com/drive/folders/16IqSkctIGp76ZYKKvJvMB_ggHcQsessM?usp=sharing), allowing you to perform inference using the provided T-DEED model. Please note that results obtained with this checkpoint may differ slightly from the reported results due to variations in different training runs.

## Acknowledgments

This repository is built upon the foundation of the [E2E-Spot](https://github.com/jhong93/spot) codebase, and we would like to extend our gratitude for their work.

## Citation

If you use this repository for your research or wish to refer to our contributions, please use the following BibTeX entries.

```bibtex
@InProceedings{Deliège2020SoccerNetv2,
      title={SoccerNet-v2 : A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos}, 
      author={Adrien Deliège and Anthony Cioppa and Silvio Giancola and Meisam J. Seikavandi and Jacob V. Dueholm and Kamal Nasrollahi and Bernard Ghanem and Thomas B. Moeslund and Marc Van Droogenbroeck},
      year={2021},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      month = {June},
}
```

```bibtex
@article{cioppa2024soccernet,
  title={SoccerNet 2023 challenges results},
  author={Cioppa, Anthony and Giancola, Silvio and Somers, Vladimir and Magera, Floriane and Zhou, Xin and Mkhallati, Hassan and Deli{\`e}ge, Adrien and Held, Jan and Hinojosa, Carlos and Mansourian, Amir M and others},
  journal={Sports Engineering},
  volume={27},
  number={2},
  pages={24},
  year={2024},
  publisher={Springer}
}
```

```bibtex
@inproceedings{xarles2024t,
  title={T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos},
  author={Xarles, Artur and Escalera, Sergio and Moeslund, Thomas B and Clap{\'e}s, Albert},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3410--3419},
  year={2024}
}
```



