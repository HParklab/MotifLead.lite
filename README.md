# first make a conda environment

> conda env create -f environment.yml 
> source active mll

# step0. prepare
- a PDB file containing receptor+ligand structure
- a mol2 file containing ligand information with atom names identical to those appear in PDB

# step1. featurize
> cd example/
> python ../featurize/featurize.py [PDB] [mol2] [ligand name in PDB]
ex) python ../featurize/featurize.py Q16769_4YWY_PBD_0_70.pdb 4YWY_PBD_0_70.mol2 LG1

# step2. run!
> python infer.py -p example -o example.npz
