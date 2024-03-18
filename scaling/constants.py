import pandas as pd

# default path to metadata about the evals
EVAL_METADATA = pd.read_csv("exp_data/eval_metadata.csv")

# 46 downstream evals and their citations used in our paper
FRIENDLY_CITATIONS = {
    "agi_eval_lsat_ar": "AGIEval LSAT AR~\cite{zhong2023agieval,zhong2019jec,Wang2021FromLT}",
    "agi_eval_lsat_rc": "AGIEval LSAT RC~\cite{zhong2023agieval,zhong2019jec,Wang2021FromLT}",
    "agi_eval_lsat_lr": "AGIEval LSAT LR~\cite{zhong2023agieval,zhong2019jec,Wang2021FromLT}",
    "agi_eval_sat_en": "AGIEval SAT English~\cite{zhong2023agieval}",
    "arc_easy": "ARC-Easy~\cite{arc}",
    "arc_challenge": "ARC-Challenge~\cite{arc}",
    "bbq": "BBQ~\cite{bbq}",
    "bigbench_conceptual_combinations": "BIG-bench: Conceptual combinations~\cite{srivastava2023beyond}",
    "bigbench_conlang_translation": "BIG-bench: Conlang translation~\cite{srivastava2023beyond}",
    "bigbench_cs_algorithms": "BIG-bench: CS algorithms~\cite{srivastava2023beyond}",
    "bigbench_dyck_languages": "BIG-bench: Dyck languages~\cite{srivastava2023beyond}",
    "bigbench_elementary_math_qa": "BIG-bench: Elementary math QA~\cite{srivastava2023beyond}",
    "bigbench_misconceptions": "BIG-bench: Misconceptions~\cite{srivastava2023beyond}",
    "bigbench_language_identification": "BIG-bench: Language identification~\cite{srivastava2023beyond}",
    "bigbench_logical_deduction": "BIG-bench: Logical deduction~\cite{srivastava2023beyond}",
    "bigbench_novel_concepts": "BIG-bench: Novel Concepts~\cite{srivastava2023beyond}",
    "bigbench_operators": "BIG-bench: Operators~\cite{srivastava2023beyond}",
    "bigbench_repeat_copy_logic": "BIG-bench: Repeat copy logic~\cite{srivastava2023beyond}",
    "bigbench_qa_wikidata": "BIG-bench: QA WikiData~\cite{srivastava2023beyond}",
    "bigbench_strange_stories": "BIG-bench: Strange stories~\cite{srivastava2023beyond}",
    "bigbench_strategy_qa": "BIG-bench: Strategy QA~\cite{srivastava2023beyond}",
    "bigbench_understanding_fables": "BIG-bench: Understanding fables~\cite{srivastava2023beyond}",
    "boolq": "BoolQ~\cite{boolq}",
    "commonsense_qa": "Commonsense QA~\cite{talmor-etal-2019-commonsenseqa}",
    "copa": "COPA~\cite{copa}",
    "coqa": "CoQA~\cite{reddy-etal-2019-coqa}",
    "enterprise_pii_classification": "Enterprise PII classification~\cite{pii}",
    "hellaswag": "HellaSwag (10-shot)~\cite{hellaswag}",
    "hellaswag_zeroshot": "HellaSwag (zero-shot)~\cite{hellaswag}",
    "jeopardy": "Jeopardy~\cite{mosaicml}",
    "lambada_openai": "LAMBADA~\cite{lambada}",
    "logi_qa": "LogiQA~\cite{Liu2020LogiQAAC}",
    "math_qa": "MathQA~\cite{mathqa}",
    "mmlu": "MMLU (5-shot)~\cite{mmlu}",
    "mmlu_zeroshot": "MMLU (zero-shot)~\cite{mmlu}",
    "openbook_qa": "OpenBook QA~\cite{OpenBookQA2018}",
    "piqa": "PIQA~\cite{piqa}",
    "pubmed_qa_labeled": "PubMed QA Labeled~\cite{pubmed}",
    "simple_arithmetic_nospaces": "Simple Arithmetic: NoSpaces~\cite{mosaicml}",
    "simple_arithmetic_withspaces": "Simple Arithmetic: WithSpaces~\cite{mosaicml}",
    "siqa": "SIQA~\cite{siqa}",
    "squad": "SQuAD~\cite{squad}",
    "winogender_mc_female": "WinoGender MC: Female~\cite{winogender}",
    "winogender_mc_male": "WinoGender MC: Male~\cite{winogender}",
    "winograd": "WinoGrand~\cite{winograd}",
    "winogrande": "WinoGrande~\cite{sakaguchi2019winogrande}",
}

# friendly names for various eval (loss and error) splits
VAL_FRIENDLIES = {
    "openlm": "OpenLM",
    "c4_val": "C4 eval",
    "val_de-en_100": "C4 German eval\n",
    "paloma_redpajama": "RedPajama (Paloma split)",
    "paloma_c4_en": "C4 (Paloma split)",
    "paloma_falcon-refinedweb": "RefinedWeb (Paloma split)",
    "paloma_dolma_100_programing_languages": "100 programming languages\n(Paloma split)",
    "paloma_ptb": "Penn Tree Bank\n(Paloma split)",
    "avg": "46-task split",
    "avg_subset": "17-task split",
}

# add for all downstream tasks
for e in FRIENDLY_CITATIONS:
    VAL_FRIENDLIES[e] = FRIENDLY_CITATIONS[e].split("~")[0]

# random baseline accuracies
random_baseline = EVAL_METADATA["Random baseline"].to_list()
eval_task = EVAL_METADATA["Eval Task"].to_list()
RANDOM_BASELINE = {eval_task[i]: random_baseline[i] / 100.0 for i in range(len(eval_task))}

RANDOM_AVG_ERROR = 0.0
for b in FRIENDLY_CITATIONS:
    RANDOM_AVG_ERROR += RANDOM_BASELINE[b]
RANDOM_AVG_ERROR = 1.0 - (RANDOM_AVG_ERROR / len(FRIENDLY_CITATIONS))

SUBSET = [
    "bigbench_operators",
    "pubmed_qa_labeled",
    "hellaswag_zeroshot",
    "boolq",
    "arc_easy",
    "coqa",
    "bigbench_dyck_languages",
    "lambada_openai",
    "bigbench_novel_concepts",
    "winograd",
    "bigbench_cs_algorithms",
    "commonsense_qa",
    "bigbench_qa_wikidata",
    "hellaswag",
    "copa",
    "squad",
    "piqa",
]

RANDOM_AVG_SUBSET_ERROR = 0.0
for b in SUBSET:
    RANDOM_AVG_SUBSET_ERROR += RANDOM_BASELINE[b]
RANDOM_AVG_SUBSET_ERROR = 1.0 - (RANDOM_AVG_SUBSET_ERROR / len(SUBSET))

SUBSET_FRIENDLY_CITATIONS = [FRIENDLY_CITATIONS[i] for i in SUBSET]


MODEL_SHAPES = {
    "d=96_l=8_h=4": "^",
    "d=512_l=8_h=4": "s",
    "d=576_l=24_h=8": "d",
    "d=1024_l=24_h=8": "p",
    "open_lm_1b": "P",
    "open_lm_7b": "*",
}

NAME_PARAMS = {
    "d=96_l=8_h=4": 10569312,
    "d=512_l=8_h=4": 78914048,
    "d=576_l=24_h=8": 153677376,
    "d=1024_l=24_h=8": 411616256,
    "open_lm_1b": 1439795200,
    "open_lm_7b": 6889410560,
}

PARAM_SHAPES = {
    10569312: "^",
    78914048: "s",
    153677376: "d",
    411616256: "p",
    1439795200: "P",
    6889410560: "*",
}

MODEL_FRINDLIES = {
    "d=96_l=8_h=4": "0.011B",
    "d=512_l=8_h=4": "0.079B",
    "d=576_l=24_h=8": "0.154B",
    "d=1024_l=24_h=8": "0.411B",
    "open_lm_1b": "1.4B",
    "open_lm_7b": "6.9B",
}

DATASET_FRIENDLIES = {
    "c4_original": "C4",
    "pile": "Pile",
    "rpj": "RedPajama",
    "rw_original": "RefinedWeb",
}

DATASET_COLORS = {
    "c4_original": "tab:blue",
    "rpj": "tab:orange",
    "rw_original": "tab:purple",
}

DATASET_SHAPES = {
    "c4_original": "s",
    "rpj": "^",
    "rw_original": "d",
}

HF_MODEL_REPO = "mlfoundations/scaling"
MODEL_JSON_ROOT = "exp_data/models"
