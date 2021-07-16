from rdkit import Chem
import pandas as pd   
import seaborn as sns
from rdkit.Chem import PandasTools 
from rdkit.Chem import Descriptors 
from rdkit.Chem import rdmolops 
import rdkit
import deepchem as dc

def get_desc_features(smiles):
  mols = [Chem.MolFromSmiles(s) for s in smiles] 
  feat = dc.feat.RDKitDescriptors() 
  arr = feat.featurize(mols) 
  return arr
#desc_features = get_desc_features(smiles_list)

def get_fingerprints(smiles):
  mols = [Chem.MolFromSmiles(smile) for smile in smiles] 
  feat = dc.feat.CircularFingerprint(size=1024) 
  arr = feat.featurize(mols) 
  return arr
#fingerprints = get_fingerprints(smiles_list)

def get_convgraph_feat(dataset):
  conv_list = []*len(final_pred_smiles)
  for i in range(len(final_pred_smiles)):
    if i % 10 == 0:
      print(i)
    obj = dataset.X[i]
    conv_list.append(obj.get_atom_features())
  return conv_list

def find_viable_smiles_QED(smiles_filepath, f_id):
  pred_smiles = pd.read_csv(smiles_filepath, header= None)
  pred_smiles.columns = ["SMILES"]
  smiles_list = pred_smiles["SMILES"].tolist()
  molecules = []
  for smile in smiles_list:
    if Chem.MolFromSmiles(smile) is not None:   
      molecules.append(smile) 
  molecules = [Chem.MolFromSmiles(x) for x in molecules] 
  qed_list = [rdkit.Chem.QED.qed(x) for x in molecules]
  final_mol_list = [(a,b) for a,b in zip(molecules,qed_list) if b > 0.5] 
  gen_smiles = [Chem.MolToSmiles(x[0]) for x in final_mol_list]
  ur_gen_smiles = list(set(gen_smiles))
  unique_smiles = ur_gen_smiles
  #print(len(ur_gen_smiles))
  all_smiles = pd.read_csv("/tmp/full.csv")
  all_smiles_list = all_smiles['SMILES'].tolist()
  common = list(set(unique_smiles).intersection(all_smiles_list))
  for com in common:
    unique_smiles.remove(com)
  print("Number of unique, non-repeating smiles in {}: {}".format(f_id,len(unique_smiles)))
  return unique_smiles

