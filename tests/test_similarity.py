import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from hsr import similarity

# Helper function to generate 3D conformer for a molecule
def generate_3d_coords(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

# Fixtures for RDKit Mol objects
@pytest.fixture
def methylamine_3d():
    mol = Chem.MolFromSmiles('CN')  
    return generate_3d_coords(mol)

@pytest.fixture
def charged_methylamine_3d():
    mol = Chem.MolFromSmiles('C[N+]')  
    return generate_3d_coords(mol)

@pytest.fixture
def ethane_3d():
    mol = Chem.MolFromSmiles('CC')
    return generate_3d_coords(mol)

def test_calculate_manhattan_distance():
    moments1 = [1, 2, 3, 4, 5]
    moments2 = [2, 3, 4, 5, 6]

    dist = similarity.calculate_manhattan_distance(moments1, moments2)
    assert dist == 5.0  # (1 + 1 + 1 + 1 + 1)

def test_calculate_similarity_from_distance():
    dist = 1
    n_components = 2
    similarity_measure = similarity.calculate_similarity_from_distance(dist, n_components)
    assert similarity_measure == 2/3  # 1 / (1 + 0.5)

# Edge Cases
def test_calculate_mean_absolute_difference_different_lengths():
    moments1 = [1, 2, 3]
    moments2 = [4, 5]
    
    with pytest.raises(IndexError):
        similarity.calculate_manhattan_distance(moments1, moments2)

def test_compute_similarity_3d_mols(methylamine_3d, ethane_3d):
    # Similarity between identical molecules
    similarity_same = similarity.compute_similarity(methylamine_3d, methylamine_3d)
    assert similarity_same == 1 

    # Similarity between different molecules
    similarity_diff_1 = similarity.compute_similarity(methylamine_3d, ethane_3d)

    assert similarity_diff_1 < 1

def test_compute_similarity_charged_vs_neutral(methylamine_3d, charged_methylamine_3d, capsys):
    similarity_score = similarity.compute_similarity(methylamine_3d, charged_methylamine_3d, chirality=True)
    assert similarity_score < 1

    captured = capsys.readouterr()
 
    expected_warning = ("WARNING: Chirality may not be consistent. 2 vectors have arbitrary signs.")
    
    assert expected_warning in captured.out    
