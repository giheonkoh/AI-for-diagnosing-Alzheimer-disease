import pandas as pd

f = ['lateralorbitofrontal_volume',
'medialorbitofrontal_volume',
'paracentral_volume',
'parsopercularis_volume',
'parsorbitalis_volume',
'parstriangularis_volume',
'precentral_volume',
'caudalmiddlefrontal_volume',
'rostralmiddlefrontal_volume',
'superiorfrontal_volume',
'frontalpole_volume']

lh_f = pd.DataFrame(f).apply(lambda r: 'lh_' + r)
rh_f = pd.DataFrame(f).apply(lambda r: 'rh_' + r)

#frontal = pd.concat([lh_f, rh_f], axis = 0)

o = ['cuneus_volume',
'lateraloccipital_volume',
'lingual_volume',
'pericalcarine_volume',
'caudalanteriorcingulate_volume',
'isthmuscingulate_volume',
'posteriorcingulate_volume',
'rostralanteriorcingulate_volume']

lh_o = pd.DataFrame(o).apply(lambda r: 'lh_' + r)
rh_o = pd.DataFrame(o).apply(lambda r: 'rh_' + r)
#occipital = pd.concat([lh_o, rh_o], axis = 0)

p = ['inferiorparietal_volume',
'postcentral_volume',
'precuneus_volume',
'superiorparietal_volume',
'supramarginal_volume']

lh_p = pd.DataFrame(p).apply(lambda r: 'lh_' + r)
rh_p = pd.DataFrame(p).apply(lambda r: 'rh_' + r)
#parietal = pd.concat([lh_o, rh_o], axis = 0)

t = ['entorhinal_volume',
'fusiform_volume',
'parahippocampal_volume',
'temporalpole_volume',
'bankssts_volume',
'inferiortemporal_volume',
'middletemporal_volume',
'superiortemporal_volume',
'transversetemporal_volume',
'insula_volume']

lh_t = pd.DataFrame(t).apply(lambda r: 'lh_' + r)
rh_t = pd.DataFrame(t).apply(lambda r: 'rh_' + r)
#temporal = pd.concat([lh_o, rh_o], axis = 0)

sgm = ['thalamus-proper',
'caudate',
'putamen',
'pallidum',
'hippocampus',
'amygdala',
'accumbens-area', 'cerebellum-white-matter','cerebellum-cortex', 'lateral-ventricle', 'inf-lat-vent']
lh_sgm = pd.DataFrame(sgm).apply(lambda r: 'left-' + r)
rh_sgm = pd.DataFrame(sgm).apply(lambda r: 'right-' + r)


right = pd.concat([rh_f, rh_o, rh_p, rh_t, rh_sgm], axis = 0)
left = pd.concat([lh_f, lh_o, lh_p, lh_t, lh_sgm], axis = 0)

csv = pd.concat([left, right], axis = 1).reset_index().iloc[:,1:]
csv.columns  = ['left', 'right']

#csv.to_csv('reportfeatures.csv', index = False)
