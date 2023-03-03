
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
import pandas as pd
import sys
from rdkit.Chem import AllChem

def read_csv_to_list(path1):
	df = pd.read_csv(path1, usecols=[1], names = None)
	df_line = df.values.tolist()
	result1 = []
	for line in df_line:
		result1.append(line[0])
	return result1

def read_csv_to_list(path2):
	df = pd.read_csv(path2, usecols=[1], names = None)
	df_line = df.values.tolist()
	result2 = []
	for line in df_line:
		result2.append(line[0])
	return result2

# SMILES to calculate ECFP4 ==================
def smiles_to_ecfp4(smiles):
	'''Use SMILES to calculate ECFP4.'''
	try:
		mol = Chem.MolFromSmiles(smiles)
		return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
	except:
		return None

# Tanimoto similarity between two fingerprints ==================
def tanimoto(fp1, fp2): 
	'''Compute Tanimoto similarity between two fingerprints'''
	return DataStructs.FingerprintSimilarity(fp1,fp2)


if __name__ == "__main__":
	path1 = 'HPK1.csv'
	path2 = 'kinase.csv'
	result1 = read_csv_to_list(path1)
	l1 = len(result1)
	result2 = read_csv_to_list(path2)
	l2 = len(result2)
	result1 = result1 * l2
	result2 = result2 * l1
	df = pd.DataFrame({'hpk1':result1, 'kinase':result2})
	df.head()
	def smile_similarity(line):
		f1 = smiles_to_ecfp4(line[0])
		f2 = smiles_to_ecfp4(line[1])
		tanimoto = DataStructs.FingerprintSimilarity(f1,f2)
		return tanimoto
	df['tanimoto'] = df.apply(smile_similarity, axis=1)
	df.to_csv('all_result.csv', mode = 'a')
	df1 = df['tanimoto']
	df1.to_csv('result.csv', mode = 'a')
	temp = list(df1)
	smiles_id = df1.iloc[temp.index(max(temp)),3]
	with open('fp_score.csv', 'w') as f:
		print(smiles_id, file = f)









     
# =============================================================================
# 	for i in range(len(result1)):
# 		fp1 = smiles_to_ecfp4(result1[i])
# 		for i in range(len(result2)):
# 			fp2 = smiles_to_ecfp4(result2[i])
# 			tanimoto = DataStructs.FingerprintSimilarity(fp1,fp2)
# 			if tanimoto == 1.0:
# 				with open ('result.csv', 'a') as f:
# 					print(result2[i], file = f)
# =============================================================================




			







# if __name__ == "__main__":
#     argv = sys.argv[1:]
#     path1 = argv[0]
#     path2 = argv[1]
#     result = read_csv_to_list(path1)
#     mol1 = path2
#     #print(result)
#  #   print(result)
# #    print(len(result))
#     i = 0
#     compare_result=[]
#     for i in range(5):
#         mol2 = result[i]
# #       print(mol2)
#         fp2 = smiles_to_ecfp4(mol2)
# #        print(fp2)
#         fp1 = smiles_to_ecfp4(mol1)
#         print(fp1)
#         results = tanimoto(fp1, fp2)
#         compare_result.append(results)
# #    print(compare_result)
    
# '''    
#     with open('Acadesine.csv','w') as f:
#         for item in compare_result:
#             new_item = ',' + ',' + ',' + str(item)
#             f.write(new_item)
#             f.write('\n')
#         f.close()
# '''
# df1 = pd.read_csv('PHK1.csv')
# data = pd.DataFrame(compare_result)
# df1['fp_score'] = data
# df1.iloc[3, 3]    #iloc obtain values from table
# df1.to_csv('new_PHK1.csv', mode = 'a')
# temp = df1['fp_score']
# temp = list(temp)
# pdb_id = df1.iloc[temp.index(max(temp)),3]
# print('you should select target {} for docking' .format(pdb_id))
# print(pdb_id)


