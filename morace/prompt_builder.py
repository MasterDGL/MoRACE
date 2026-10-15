import argparse
import hashlib
import json
import os
from typing import Dict

PROMPT_VERSION = "dccd_prompt_planA_v1"
LEGACY_SIGNATURES = [
    "Calculate",
    "Assumed you are",
    "Assume you are a very experienced Chemist",
]


BASE_TASK_DESCRIPTIONS = {
    "bbbp": "whether a molecule is blood-brain barrier permeable",
    "bace": "whether a molecule inhibits human beta-secretase 1 (BACE-1)",
    "clintox": "whether a molecule is associated with clinical toxicity or failure due to toxicity",
    "hiv": "whether a molecule inhibits HIV replication",
    "esol": "aqueous solubility (log solubility in mol/L) of a molecule",
    "freesolv": "hydration free energy of a molecule in water",
    "lipophilicity": "octanol/water distribution coefficient (logD at pH 7.4) of a molecule",
}

TOX21_TASK_DESCRIPTIONS = {
    "nr-ar": "toxicity activity against the androgen receptor in the nuclear receptor pathway",
    "nr-ar-lbd": "toxicity activity against the androgen receptor ligand-binding domain in the nuclear receptor pathway",
    "nr-ahr": "toxicity activity against the aryl hydrocarbon receptor in the nuclear receptor pathway",
    "nr-aromatase": "toxicity activity against aromatase in the nuclear receptor pathway",
    "nr-er": "toxicity activity against the estrogen receptor in the nuclear receptor pathway",
    "nr-er-lbd": "toxicity activity against the estrogen receptor ligand-binding domain in the nuclear receptor pathway",
    "nr-ppar-gamma": "toxicity activity against peroxisome proliferator-activated receptor gamma in the nuclear receptor pathway",
    "sr-are": "toxicity activity against the antioxidant response element in the stress response pathway",
    "sr-atad5": "genotoxicity activity indicated by ATAD5 in the stress response pathway",
    "sr-hse": "toxicity activity against the heat shock response element in the stress response pathway",
    "sr-mmp": "toxicity activity against mitochondrial membrane potential in the stress response pathway",
    "sr-p53": "toxicity activity against the DNA-damage p53 pathway in the stress response pathway",
}

SIDER_TASK_DESCRIPTIONS = {
    "cardiac disorders": "whether a molecule is associated with cardiac disorders",
    "neoplasms benign, malignant and unspecified (incl cysts and polyps)": "whether a molecule is associated with benign or malignant neoplasms",
    "hepatobiliary disorders": "whether a molecule is associated with hepatobiliary disorders",
    "metabolism and nutrition disorders": "whether a molecule is associated with metabolism and nutrition disorders",
    "product issues": "whether a molecule is associated with product issues",
    "eye disorders": "whether a molecule is associated with eye disorders",
    "investigations": "whether a molecule is associated with investigation abnormalities",
    "musculoskeletal and connective tissue disorders": "whether a molecule is associated with musculoskeletal and connective tissue disorders",
    "gastrointestinal disorders": "whether a molecule is associated with gastrointestinal disorders",
    "social circumstances": "whether a molecule is associated with social-circumstance adverse events",
    "immune system disorders": "whether a molecule is associated with immune-system disorders",
    "reproductive system and breast disorders": "whether a molecule is associated with reproductive or breast disorders",
    "general disorders and administration site conditions": "whether a molecule is associated with general disorders or administration-site conditions",
    "endocrine disorders": "whether a molecule is associated with endocrine disorders",
    "surgical and medical procedures": "whether a molecule is associated with surgical or medical procedure-related adverse events",
    "vascular disorders": "whether a molecule is associated with vascular disorders",
    "blood and lymphatic system disorders": "whether a molecule is associated with blood or lymphatic system disorders",
    "skin and subcutaneous tissue disorders": "whether a molecule is associated with skin and subcutaneous tissue disorders",
    "congenital, familial and genetic disorders": "whether a molecule is associated with congenital, familial, or genetic disorders",
    "infections and infestations": "whether a molecule is associated with infections and infestations",
    "respiratory, thoracic and mediastinal disorders": "whether a molecule is associated with respiratory, thoracic, or mediastinal disorders",
    "psychiatric disorders": "whether a molecule is associated with psychiatric disorders",
    "renal and urinary disorders": "whether a molecule is associated with renal or urinary disorders",
    "pregnancy, puerperium and perinatal conditions": "whether a molecule is associated with pregnancy or perinatal conditions",
    "ear and labyrinth disorders": "whether a molecule is associated with ear or labyrinth disorders",
    "nervous system disorders": "whether a molecule is associated with nervous-system disorders",
    "injury, poisoning and procedural complications": "whether a molecule is associated with injury, poisoning, or procedural complications",
}

QM9_TASK_DESCRIPTIONS = {
    "mu": "dipole moment of a molecule without using 3D coordinates",
    "alpha": "isotropic polarizability of a molecule without using 3D coordinates",
    "R^2": "electronic spatial extent (R^2) of a molecule without using 3D coordinates",
    "epsilon_HOMO": "highest occupied molecular orbital energy of a molecule without using 3D coordinates",
    "epsilon_LUMO": "lowest unoccupied molecular orbital energy of a molecule without using 3D coordinates",
    "Delta_epsilon": "HOMO-LUMO gap of a molecule without using 3D coordinates",
    "ZPVE": "zero-point vibrational energy of a molecule without using 3D coordinates",
    "U_0": "internal energy at 0 Kelvin of a molecule without using 3D coordinates",
    "U": "internal energy at 298.15 Kelvin of a molecule without using 3D coordinates",
    "H": "enthalpy at 298.15 Kelvin of a molecule without using 3D coordinates",
    "G": "Gibbs free energy at 298.15 Kelvin of a molecule without using 3D coordinates",
    "c_v": "heat capacity at constant volume of a molecule without using 3D coordinates",
}


def build_synthesize_prompt(task_desc: str, rule_budget: int) -> str:
    return (
        f"You are preparing the Stage 1 chemistry draft for molecular property prediction. "
        f"The target is {task_desc}. Propose about {rule_budget} candidate determinants that a medicinal chemist "
        "would seriously inspect. Focus on descriptors, functional groups, ring systems, ionization, polarity, "
        "lipophilicity, hydrogen-bonding, aromaticity, rigidity, and countable motifs when relevant. "
        "For each determinant, mention the likely directionality and optionally a threshold only when chemically justified. "
        "Write draft notes that can later be projected by the same model into structured rule objects. "
        "Try to cover at least two feature families when the task supports it, such as descriptor plus motif/ring/count. "
        "Write concise free-text chemistry notes, not JSON, not code, and avoid checklist-style descriptor dumps. "
        "Use the format 'Determinant 1:', 'Determinant 2:' and keep each item to 1-2 sentences."
    )


def build_inference_prompt(task_desc: str) -> str:
    return (
        f"You are preparing the Stage 1 data-driven chemistry draft for molecular property prediction. "
        f"The target is {task_desc}. You will receive representative labeled training molecules and a short "
        "descriptor summary. Infer 2-3 batch-level candidate determinants that may explain the target. "
        "Do not copy SMILES, labels, or training examples into the answer. Do not continue the examples. "
        "Abstract them into chemistry. Prefer descriptors, motifs, ring systems, ionization state, polarity, "
        "lipophilicity, hydrogen-bonding, and countable motifs. Do not repeat scaffold words such as Instruction, "
        "Example, Citation, or Question. If the batch-level evidence points to motifs or ring systems, do not force "
        "everything into a MolLogP or molecular-weight template. If a threshold is uncertain, say it is optional. "
        "These notes will later be projected into rule objects by the same modeling pipeline, so preserve the draft semantics. "
        "Respond in plain text using this format: 'Determinant A:', 'Direction:', 'Rationale:'. Do not output JSON."
    )


def get_synthesize_task_prompt() -> Dict:
    prompt_dict: Dict = {}
    for dataset, desc in BASE_TASK_DESCRIPTIONS.items():
        prompt_dict[f"{dataset}_small"] = build_synthesize_prompt(desc, 8)
        prompt_dict[f"{dataset}_big"] = build_synthesize_prompt(desc, 12)

    prompt_dict["tox21_small"] = {
        key: build_synthesize_prompt(desc, 8)
        for key, desc in TOX21_TASK_DESCRIPTIONS.items()
    }
    prompt_dict["tox21_big"] = {
        key: build_synthesize_prompt(desc, 12)
        for key, desc in TOX21_TASK_DESCRIPTIONS.items()
    }
    prompt_dict["sider_small"] = {
        key: build_synthesize_prompt(desc, 8)
        for key, desc in SIDER_TASK_DESCRIPTIONS.items()
    }
    prompt_dict["sider_big"] = {
        key: build_synthesize_prompt(desc, 12)
        for key, desc in SIDER_TASK_DESCRIPTIONS.items()
    }

    for subtask, desc in QM9_TASK_DESCRIPTIONS.items():
        prompt_dict[f"{subtask}_small"] = build_synthesize_prompt(desc, 8)
        prompt_dict[f"{subtask}_big"] = build_synthesize_prompt(desc, 12)
    return prompt_dict


def get_inference_task_prompt() -> Dict:
    prompt_dict: Dict = {}
    for dataset, desc in BASE_TASK_DESCRIPTIONS.items():
        prompt_dict[dataset] = build_inference_prompt(desc)
    prompt_dict["tox21"] = {
        key: build_inference_prompt(desc)
        for key, desc in TOX21_TASK_DESCRIPTIONS.items()
    }
    prompt_dict["sider"] = {
        key: build_inference_prompt(desc)
        for key, desc in SIDER_TASK_DESCRIPTIONS.items()
    }
    for subtask, desc in QM9_TASK_DESCRIPTIONS.items():
        prompt_dict[subtask] = build_inference_prompt(desc)
    return prompt_dict


def write_prompt_metadata(output_path: str, task: str, payload: Dict) -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    meta = {
        "task": task,
        "prompt_version": PROMPT_VERSION,
        "output_file": os.path.basename(output_path),
        "sha256": digest,
        "legacy_signatures": LEGACY_SIGNATURES,
    }
    meta_path = output_path.replace(".json", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def main():
    os.makedirs(args.output_folder, exist_ok=True)
    if args.task == "synthesize":
        payload = get_synthesize_task_prompt()
        output_path = os.path.join(args.output_folder, "synthesize_prompt.json")
    elif args.task == "inference":
        payload = get_inference_task_prompt()
        output_path = os.path.join(args.output_folder, "inference_prompt.json")
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_prompt_metadata(output_path, args.task, payload)
    print(f"Wrote {args.task} prompts to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["synthesize", "inference"])
    parser.add_argument("--output_folder", type=str, default="prompt_file")
    args = parser.parse_args()
    main()
