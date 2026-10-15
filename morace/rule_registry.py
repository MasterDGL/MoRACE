from typing import Callable, Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors


DESCRIPTOR_REGISTRY: Dict[str, Callable] = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHBD": rdMolDescriptors.CalcNumHBD,
    "NumHBA": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "LabuteASA": MolSurf.LabuteASA,
    "ExactMolWt": Descriptors.ExactMolWt,
    "MolMR": Crippen.MolMR,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Chi0": Descriptors.Chi0,
    "Chi1": Descriptors.Chi1,
}

DESCRIPTOR_CODE_REGISTRY: Dict[str, str] = {
    "MolWt": "Descriptors.MolWt(mol)",
    "MolLogP": "Crippen.MolLogP(mol)",
    "TPSA": "rdMolDescriptors.CalcTPSA(mol)",
    "NumHBD": "rdMolDescriptors.CalcNumHBD(mol)",
    "NumHBA": "rdMolDescriptors.CalcNumHBA(mol)",
    "NumRotatableBonds": "Lipinski.NumRotatableBonds(mol)",
    "NumAromaticRings": "rdMolDescriptors.CalcNumAromaticRings(mol)",
    "NumAliphaticRings": "rdMolDescriptors.CalcNumAliphaticRings(mol)",
    "RingCount": "rdMolDescriptors.CalcNumRings(mol)",
    "HeavyAtomCount": "Descriptors.HeavyAtomCount(mol)",
    "FractionCSP3": "rdMolDescriptors.CalcFractionCSP3(mol)",
    "NumHeteroatoms": "Lipinski.NumHeteroatoms(mol)",
    "LabuteASA": "MolSurf.LabuteASA(mol)",
    "ExactMolWt": "Descriptors.ExactMolWt(mol)",
    "MolMR": "Crippen.MolMR(mol)",
    "NumValenceElectrons": "Descriptors.NumValenceElectrons(mol)",
    "NumSaturatedRings": "rdMolDescriptors.CalcNumSaturatedRings(mol)",
    "NHOHCount": "Lipinski.NHOHCount(mol)",
    "NOCount": "Lipinski.NOCount(mol)",
    "BalabanJ": "Descriptors.BalabanJ(mol)",
    "BertzCT": "Descriptors.BertzCT(mol)",
    "HallKierAlpha": "Descriptors.HallKierAlpha(mol)",
    "Chi0": "Descriptors.Chi0(mol)",
    "Chi1": "Descriptors.Chi1(mol)",
}


def _has_fused_ring(mol) -> bool:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    for idx, left in enumerate(rings):
        for right in rings[idx + 1 :]:
            if len(left & right) >= 2:
                return True
    return False


def _has_aromatic_ring(mol) -> bool:
    return rdMolDescriptors.CalcNumAromaticRings(mol) > 0


CUSTOM_PATTERN_FUNCTIONS: Dict[str, Callable] = {
    "aromatic_ring": lambda mol: int(_has_aromatic_ring(mol)),
    "fused_ring": lambda mol: int(_has_fused_ring(mol)),
}

CUSTOM_PATTERN_CODE_REGISTRY: Dict[str, str] = {
    "aromatic_ring": "int(rdMolDescriptors.CalcNumAromaticRings(mol) > 0)",
    "fused_ring": "int(_has_fused_ring(mol))",
}

PATTERN_ALIAS: Dict[str, str] = {
    "halogen": "[F,Cl,Br,I]",
    "nitro_group": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
    "alcohol": "[OX2H]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "ether": "[OD2]([#6])[#6]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "sulfone": "[SX4](=O)(=O)([#6])[#6]",
    "nitrile": "[CX2]#N",
    "heteroaromatic_ring": "[a;r5,r6]1[a,n,o,s][a,n,o,s][a,n,o,s][a,n,o,s][a,n,o,s]1",
    "benzene_ring": "c1ccccc1",
    "phenyl": "c1ccccc1",
    "hydroxyl": "[OX2H]",
    "heterocycle": "[R;!#6]",
}

COUNT_ALIAS: Dict[str, Callable] = {
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumHBD": rdMolDescriptors.CalcNumHBD,
    "NumHBA": rdMolDescriptors.CalcNumHBA,
}

FEATURE_ALIASES: Dict[str, str] = {
    "molecular weight": "MolWt",
    "molecular_weight": "MolWt",
    "mol weight": "MolWt",
    "mol. weight": "MolWt",
    "molweight": "MolWt",
    "molecular mass": "MolWt",
    "mw": "MolWt",
    "molwt": "MolWt",
    "mol_wt": "MolWt",
    "logp": "MolLogP",
    "c log p": "MolLogP",
    "c_log_p": "MolLogP",
    "clogp": "MolLogP",
    "mol logp": "MolLogP",
    "mol_logp": "MolLogP",
    "partition coefficient": "MolLogP",
    "topological polar surface area": "TPSA",
    "polar surface area": "TPSA",
    "polar_surface_area": "TPSA",
    "psa": "TPSA",
    "tpsa": "TPSA",
    "hydrogen bond donor": "NumHBD",
    "hydrogen bond donors": "NumHBD",
    "h-bond donor": "NumHBD",
    "h bond donors": "NumHBD",
    "hbd": "NumHBD",
    "numhbd": "NumHBD",
    "num_hbd": "NumHBD",
    "h bond donor": "NumHBD",
    "hydrogen bond acceptor": "NumHBA",
    "hydrogen bond acceptors": "NumHBA",
    "h-bond acceptor": "NumHBA",
    "h bond acceptor": "NumHBA",
    "h bond acceptors": "NumHBA",
    "hba": "NumHBA",
    "numhba": "NumHBA",
    "num_hba": "NumHBA",
    "rotatable bonds": "NumRotatableBonds",
    "rotatable bond": "NumRotatableBonds",
    "num rotatable bonds": "NumRotatableBonds",
    "num_rotatable_bonds": "NumRotatableBonds",
    "numrotatablebonds": "NumRotatableBonds",
    "aromatic rings": "NumAromaticRings",
    "aromatic ring count": "NumAromaticRings",
    "num aromatic rings": "NumAromaticRings",
    "num_aromatic_rings": "NumAromaticRings",
    "numaromaticrings": "NumAromaticRings",
    "aliphatic rings": "NumAliphaticRings",
    "ring count": "RingCount",
    "ring_count": "RingCount",
    "rings": "RingCount",
    "heavy atoms": "HeavyAtomCount",
    "heavy atom count": "HeavyAtomCount",
    "heavy_atom_count": "HeavyAtomCount",
    "fraction csp3": "FractionCSP3",
    "fraction_csp3": "FractionCSP3",
    "fcsp3": "FractionCSP3",
    "heteroatoms": "NumHeteroatoms",
    "heteroatom count": "NumHeteroatoms",
    "labute asa": "LabuteASA",
    "exact molecular weight": "ExactMolWt",
    "molar refractivity": "MolMR",
    "valence electrons": "NumValenceElectrons",
    "saturated rings": "NumSaturatedRings",
    "nhohcount": "NHOHCount",
    "nocount": "NOCount",
    "balabanj": "BalabanJ",
    "bertzct": "BertzCT",
    "hall kier alpha": "HallKierAlpha",
    "chi0": "Chi0",
    "chi1": "Chi1",
}

PATTERN_ALIASES: Dict[str, str] = {
    "halogens": "halogen",
    "halogenated": "halogen",
    "halogen substituent": "halogen",
    "halogen substituents": "halogen",
    "fluorine": "halogen",
    "chlorine": "halogen",
    "bromine": "halogen",
    "nitro": "nitro_group",
    "nitro group": "nitro_group",
    "nitro_group": "nitro_group",
    "hydroxyl": "alcohol",
    "hydroxyl group": "alcohol",
    "amines": "amine",
    "amino": "amine",
    "basic nitrogen": "amine",
    "tertiary amine": "amine",
    "amides": "amide",
    "amide group": "amide",
    "amide bond": "amide",
    "carbonyl group": "carbonyl",
    "carboxylic acid": "carboxyl",
    "carboxyl group": "carboxyl",
    "esters": "ester",
    "ethers": "ether",
    "sulfonamides": "sulfonamide",
    "sulfones": "sulfone",
    "cyano": "nitrile",
    "nitriles": "nitrile",
    "heteroaromatic": "heteroaromatic_ring",
    "heteroaromatic ring": "heteroaromatic_ring",
    "heteroaromatic_ring": "heteroaromatic_ring",
    "hetero-aromatic ring": "heteroaromatic_ring",
    "fused rings": "fused_ring",
    "fused bicyclic ring": "fused_ring",
    "fused_ring": "fused_ring",
    "fused ring system": "fused_ring",
    "aromatic ring": "aromatic_ring",
    "aromatic_ring": "aromatic_ring",
    "aromatic rings": "aromatic_ring",
    "benzene": "benzene_ring",
    "phenyl ring": "phenyl",
    "heterocyclic": "heterocycle",
    "heterocycle ring": "heterocycle",
}


def canonicalize_feature_name(feature_family: str, feature_name: str) -> Optional[str]:
    raw = (feature_name or "").strip()
    if not raw:
        return None
    lowered = " ".join(raw.replace("-", " ").replace("_", " ").split()).lower()
    compact = lowered.replace(" ", "")
    if feature_family == "descriptor":
        if raw in DESCRIPTOR_REGISTRY:
            return raw
        return FEATURE_ALIASES.get(lowered) or FEATURE_ALIASES.get(compact)
    if feature_family == "count":
        if raw in COUNT_ALIAS:
            return raw
        return FEATURE_ALIASES.get(lowered) or FEATURE_ALIASES.get(compact) or PATTERN_ALIASES.get(lowered) or PATTERN_ALIASES.get(compact)
    if feature_family == "pattern_presence":
        if raw in PATTERN_ALIAS or raw in CUSTOM_PATTERN_FUNCTIONS:
            return raw
        return PATTERN_ALIASES.get(lowered) or PATTERN_ALIASES.get(compact)
    return None


def get_smarts(feature_name: str) -> Optional[str]:
    return PATTERN_ALIAS.get(feature_name)


def get_compiled_pattern(feature_name: str):
    smarts = get_smarts(feature_name)
    if not smarts:
        return None
    return Chem.MolFromSmarts(smarts)


def get_pattern_callable(feature_name: str) -> Optional[Callable]:
    if feature_name in CUSTOM_PATTERN_FUNCTIONS:
        return CUSTOM_PATTERN_FUNCTIONS[feature_name]
    compiled = get_compiled_pattern(feature_name)
    if compiled is None:
        return None
    return lambda mol, patt=compiled: int(bool(mol.HasSubstructMatch(patt)))


def get_pattern_code(feature_name: str) -> Optional[str]:
    if feature_name in CUSTOM_PATTERN_CODE_REGISTRY:
        return CUSTOM_PATTERN_CODE_REGISTRY[feature_name]
    smarts = get_smarts(feature_name)
    if not smarts:
        return None
    return f"int(bool(mol.HasSubstructMatch(Chem.MolFromSmarts({smarts!r}))))"


def is_custom_pattern(feature_name: str) -> bool:
    return feature_name in CUSTOM_PATTERN_FUNCTIONS
