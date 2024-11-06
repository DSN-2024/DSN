# DSN Jailbreak Attack & Evaluation Ensemble

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for [`Don't Say No: Jailbreaking LLM by Suppressing Refusal`](https://arxiv.org/abs/2404.16369)

In this work, we propose one novel jailbreak optmization target, introducing the refusal loss, cosine decay weighting schedule method, applying the Unlikelihood loss to ensure convergence, and novelly incorporate a NLI based method into our proposed jailbreak Evaluation Ensemble pipeline.

<p align="center">
  <img src="main.png" width="70%">
</p>

## Installation

Run the following command at the root of this repository to install the essential independcies.

```bash
pip install -e .
python -m spacy download en_core_web_sm
```
Note that we have chosen a different `transformers` version from the default `GCG` implementation, which might introduce subtle difference in Llama-2 model conversation prompt formatting. To faithfully reproduce all the results reported in the paper, e.g. both [`DSN`](https://arxiv.org/abs/2404.16369) and [`GCG`](https://arxiv.org/abs/2307.15043) attack results upon Llama-2 model, try install the `transformers` package with version 4.28.1 and `fschat` package with version 0.2.20. Both version of conda environmnets are supported by our implementation.

## To reproduce

First, download all the target models as well as the NLI and $\textit{HarmBench}$ model utilized in Evaluation Ensemble, and configure the OpenAI client in file `eval_ensemble_src/eval2_gpt.py` properly.

The attack launch file is located as `experiments/launch_scripts/run_dsn_attack.sh`

The recommend hyper-parameter setting is already detailed in the script. For rigorous evaluation, please consider running the same setting for multiple times.

To evaluate, please first run the scripts in eval_scripts folder, e.g. `experiments/eval_scripts/lastStep_eval.sh`, to conduct evaluation under the Refusal Matching metric. Then, change directory to eval_ensemble_src to launch `eval_ensemble_src/launch_majority_eval.sh`

## Updates

- 2024.11.6:  Refine the _update_ids function, to support llama-2 conv_template within different `transformers` and `fschat` version. For more details please refer to Section B.1.1 of our paper

## Citation
If you find our implementation and paper useful, please consider citing our work:

```bibtex
@misc{zhou2024dontsaynojailbreaking,
      title={Don't Say No: Jailbreaking LLM by Suppressing Refusal}, 
      author={Yukai Zhou and Wenjie Wang},
      year={2024},
      eprint={2404.16369},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.16369}, 
}
```

In addition, please also consider citing the following works, which have been utilized in this paper

```bibtex
@misc{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
  year={2023},
  eprint={2307.15043},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2402.04249},
  year={2024}
}

@inproceedings{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Patrick Chao and Edoardo Debenedetti and Alexander Robey and Maksym Andriushchenko and Francesco Croce and Vikash Sehwag and Edgar Dobriban and Nicolas Flammarion and George J. Pappas and Florian Tramèr and Hamed Hassani and Eric Wong},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}

@article{huang2023catastrophic,
  title={Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation},
  author={Huang, Yangsibo and Gupta, Samyak and Xia, Mengzhou and Li, Kai and Chen, Danqi},
  journal={arXiv preprint arXiv:2310.06987},
  year={2023}
}
```

## Acknowledgment
`DSN` attack implementation is built upon modifying the `GCG` implementaion from [GCG github repo](https://github.com/llm-attacks/llm-attacks)

Sincere thanks to the authors of [Zou et al. 2023](https://arxiv.org/abs/2307.15043) for the valueable insight and open-sourced code.

## License
`DSN` is licensed under the terms of the MIT license. See LICENSE for more details.