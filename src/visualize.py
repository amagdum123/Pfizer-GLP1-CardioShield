from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def get_highlight_atoms(smiles, important_bits):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")


    bit_info = {}

    AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
        bitInfo=bit_info
    )


    highlight_atoms = []


    for bit in important_bits:

        if bit in bit_info:

            for atom_radius in bit_info[bit]:

                atom_id = atom_radius[0]

                highlight_atoms.append(atom_id)


    return mol, list(set(highlight_atoms))



def draw_molecule(smiles, important_bits):

    mol, atoms = get_highlight_atoms(
        smiles,
        important_bits
    )


    drawer = rdMolDraw2D.MolDraw2DCairo(
        500,
        500
    )


    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms
    )


    drawer.FinishDrawing()


    png = drawer.GetDrawingText()


    with open(
        "results/molecule_explanation.png",
        "wb"
    ) as f:

        f.write(png)


    print(
        "Saved molecule visualization!"
    )



if __name__ == "__main__":


    smiles = input(
        "Enter SMILES: "
    )


    # Example important Morgan bits
    # These will later come from SHAP automatically

    important_bits = [
        807,
        80,
        309,
        1238
    ]


    draw_molecule(
        smiles,
        important_bits
    )