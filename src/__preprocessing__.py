
import pandas as pd

def getdata(file, sheet_name = None) :
    data = pd.read_csv(file)
    return data

def con(dat1, dat2) :
    return pd.concat([dat1, dat2], axis = 0)

def filtercolumns(data) :
    delete = ['Unnamed: 0','id','brain-stem','optic-chiasm','left-cerebellum-cortex','right-cerebellum-cortex','left-cerebellum-white-matter', 'right-cerebellum-white-matter','csf', 'left-choroid-plexus','right-choroid-plexus','left-vessel','right-vessel']
    delete = delete + ['wm-hypointensities','left-wm-hypointensities','right-wm-hypointensities','non-wm-hypointensities', 'left-non-wm-hypointensities','right-non-wm-hypointensities','left-thalamus', 'right-thalamus']
    delete = delete + ['brainsegvolnotvent_y.2', 'brainsegvolnotvent_x.1', 'brainsegvolnotvent_y', 'brainsegvolnotvent_y.1', 'brainsegvolnotvent_x.2', 'brainsegvolnotvent_x.3', 'brainsegvolnotvent_y.3']
    tokeep = [col for col in data.columns.tolist() if col not in delete]
    return data[tokeep]

def FeatureSelection(data, prob = 0.05) : #On call Only
    from scipy import stats

    t0 = data[data['label'] == 0]
    t1 = data[data['label'] == 1]
    features = []
    t_val = []

    for col in t0.columns.tolist() :
        if col not in ['label','id']:
            if len(t0) < len(t1) :
                test0 = t0[col]
                test1 = t1[col][:len(t0)]
            elif len(t0) > len(t1) :
                test0 = t0[col][:len(t1)]
                test1 = t1[col]
            else :
                test0 = t0[col]
                test1 = t1[col]

            ttest = stats.ttest_rel(test0, test1)

            features.append(col)
            t_val.append(ttest[0])

    df = {'features': features, 't_values':t_val}
    df = pd.DataFrame(df)
    df = df.sort_values(by = ['t_values'], ascending = False).reset_index()

    edgeplus = df['t_values'].tolist()[0]
    edgeminus = df['t_values'].tolist()[-1]

    tokeep = df[(df['t_values'] < -(abs(edgeplus) + abs(edgeminus)) * prob) | (df['t_values'] > (abs(edgeplus) + abs(edgeminus)) * prob)]['features'].tolist()
    delete = df[(df['t_values'] >= -(abs(edgeplus) + abs(edgeminus)) * prob) & (df['t_values'] <= (abs(edgeplus) + abs(edgeminus)) * prob)]['features']
    print("\n\n!!! WARNING !!!\n!!!", len(delete), "features are deleted. !!!")
    print("!!! {} out of {} features are selected. !!!\n".format(len(tokeep), len(df['features'].tolist())))
    tokeep = tokeep + ['y']
    return data[tokeep]

def normalization(x, how = 'minmax') :
    if how == 'minmax' or how == 'mm':
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
    elif how == 'standard' or how == 'std':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
    return sc.fit_transform(x)

def xysplit(dat) :
    x = dat.drop('label', axis = 1).to_numpy()
    y = dat['label']
    return x,y

def split_test_train(X,y, test_size = 0.3) :
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(X ,y, test_size = test_size, random_state = 42)
    return xtrain,xtest,ytrain,ytest

def checkpoint() :
    return print('\n\n!! Pinned Up Here !!\n\n')

def saveColumns(list, csvfile) :
    import csv
    with open(csvfile, "w") as o :
        writer = csv.writer(o, lineterminator = '\n')
        for val in list :
            writer.writerow([val])

def getColumns(csvfile) :
    list = []
    import csv
    import numpy as np
    with open(csvfile, "r") as r :
        rdr = csv.reader(r)
        for val in rdr :
            list.append(val)
    return np.array(list).flatten()

def merge_data(uploadPath, inputdir) :
    import os
    os.chdir(uploadPath)

    asegvol = pd.read_table(inputdir + '_aseg.vol.table', sep = '\t') #Measure:volume
    lh_aparc_area= pd.read_table(inputdir + '_aparc.lh.area.table', sep = '\t') #lh.aparc.area
    lh_aparc_thickness = pd.read_table(inputdir + '_aparc.lh.thick.table', sep = '\t') #lh.aparc.thickness
    lh_aparc_vol = pd.read_table(inputdir + '_aparc.lh.vol.table', sep = '\t') #lh.aparc.volume
    rh_aparc_area = pd.read_table(inputdir + '_aparc.rh.area.table', sep = '\t') #rh.aparc.area
    rh_aparc_thickness = pd.read_table(inputdir + '_aparc.rh.thick.table', sep = '\t') #rh.aparc.thickness
    rh_aparc_vol = pd.read_table(inputdir + '_aparc.rh.vol.table', sep = '\t') #rh.aparc.volume
    lh_aparc_meancurv = pd.read_table(inputdir + '_aparc.lh.meancurv.table', sep = '\t') #lh.aparc.meancurv
    rh_aparc_meancurv = pd.read_table(inputdir + '_aparc.rh.meancurv.table', sep = '\t') #rh.aparc.meancurv

    dat = pd.merge(asegvol,lh_aparc_area,left_on= 'Measure:mean', right_on='lh.aparc.area', how='inner')
    dat = pd.merge(dat,lh_aparc_thickness,left_on= 'Measure:mean', right_on='lh.aparc.thickness', how='inner')
    dat = pd.merge(dat,lh_aparc_vol,left_on= 'Measure:mean', right_on='lh.aparc.volume', how='inner')
    dat = pd.merge(dat,rh_aparc_area,left_on= 'Measure:mean', right_on='rh.aparc.area', how='inner')
    dat = pd.merge(dat,rh_aparc_thickness,left_on= 'Measure:mean', right_on='rh.aparc.thickness', how='inner')
    dat = pd.merge(dat,rh_aparc_vol,left_on= 'Measure:mean', right_on='rh.aparc.volume', how='inner')
    dat = pd.merge(dat,lh_aparc_meancurv,left_on= 'Measure:mean', right_on='lh.aparc.meancurv', how='inner')
    dat = pd.merge(dat,rh_aparc_meancurv,left_on= 'Measure:mean', right_on='rh.aparc.meancurv', how='inner')

    dat['Measure:mean'] = dat['Measure:mean'].map(lambda r: r.replace('.nii',''))
    dat.columns = dat.columns.map(lambda r:r.lower())
    dat = dat.drop(['lh.aparc.area','lh.aparc.thickness','lh.aparc.volume','rh.aparc.area','rh.aparc.thickness','rh.aparc.volume','lh.aparc.meancurv','rh.aparc.meancurv'], axis = 1)
    dat.columns = ['id'] + dat.columns.tolist()[1:]

    return dat.to_csv('{}_data.csv'.format(inputdir), index = False)


def reportinfo(inputdir, inputdir_data, mri, result) :
    #volumn data
    data = pd.read_csv(inputdir_data)
    left_features = pd.read_csv('reportfeatures.csv')['left'].T.tolist()
    right_features = pd.read_csv('reportfeatures.csv')['right'].T.tolist()
    lh = data[left_features]
    rh = data[right_features]

    lh.columns = pd.read_csv('reportfeatures.csv')['left'].map(lambda r: '_'.join(r.replace("-","_",1).split('_')[1:])).tolist()
    rh.columns = lh.columns
    d = pd.concat([lh, rh], axis = 0).T
    d['total'] = d.iloc[:,0] + d.iloc[:,1]
    d.columns = ['left', 'right', 'total']

    total_frontal_lobe = [sum(d.loc['lateralorbitofrontal_volume':'frontalpole_volume','left'].tolist()), sum(d.loc['lateralorbitofrontal_volume':'cuneus_volume','right'].tolist()), sum(d.loc['lateralorbitofrontal_volume':'cuneus_volume','left'].tolist()) + sum(d.loc['lateralorbitofrontal_volume':'cuneus_volume','right'].tolist())]
    total_cingulate_cortex = [sum(d.loc['caudalanteriorcingulate_volume':'rostralanteriorcingulate_volume','left'].tolist()), sum(d.loc['caudalanteriorcingulate_volume':'rostralanteriorcingulate_volume','right'].tolist()), sum(d.loc['caudalanteriorcingulate_volume':'rostralanteriorcingulate_volume','left'].tolist()) + sum(d.loc['caudalanteriorcingulate_volume':'rostralanteriorcingulate_volume','right'].tolist())]
    total_occipital_lobe = [sum(d.loc['cuneus_volume':'rostralanteriorcingulate_volume','left'].tolist()), sum(d.loc['cuneus_volume':'rostralanteriorcingulate_volume','right'].tolist()), sum(d.loc['cuneus_volume':'rostralanteriorcingulate_volume','left'].tolist()) + sum(d.loc['cuneus_volume':'rostralanteriorcingulate_volume','right'].tolist())]
    total_parietal_lobe = [sum(d.loc['inferiorparietal_volume':'supramarginal_volume','left'].tolist()), sum(d.loc['inferiorparietal_volume':'supramarginal_volume','right'].tolist()), sum(d.loc['inferiorparietal_volume':'supramarginal_volume','left'].tolist()) + sum(d.loc['inferiorparietal_volume':'supramarginal_volume','right'].tolist())]
    total_temporal_lobe = [sum(d.loc['entorhinal_volume':'transversetemporal_volume','left'].tolist()), sum(d.loc['entorhinal_volume':'transversetemporal_volume','right'].tolist()), sum(d.loc['entorhinal_volume':'transversetemporal_volume','left'].tolist()) + sum(d.loc['entorhinal_volume':'transversetemporal_volume','right'].tolist())]
    total_cortex_vol = [sum(x) for x in zip(total_frontal_lobe, total_occipital_lobe, total_parietal_lobe, total_temporal_lobe)]
    total_subcortical_gm = [sum(d.loc['thalamus-proper':'accumbens-area','left'].tolist()), sum(d.loc['thalamus-proper':'accumbens-area','right'].tolist()), sum(d.loc['thalamus-proper':'accumbens-area','left'].tolist()) + sum(d.loc['thalamus-proper':'accumbens-area','right'].tolist())]

    d.loc['total_frontal_lobe', :] = total_frontal_lobe
    d.loc['total_cingulate_cortex',:] = total_cingulate_cortex
    d.loc['total_occipital_lobe', :] = total_occipital_lobe
    d.loc['total_parietal_lobe', :] = total_parietal_lobe
    d.loc['total_temporal_lobe', :] = total_temporal_lobe
    d.loc['total_cortex_volume', :] = total_cortex_vol
    d.loc['total_subcortical_gm',: ] = total_subcortical_gm

    dd = d.reset_index()

    cc_vent = ['cc_anterior', 'cc_mid_anterior', 'cc_central', 'cc_mid_posterior', 'cc_posterior','3rd-ventricle','4th-ventricle']
    corpus_vent = data[cc_vent].T
    corpus_vent.columns = ['corpus_collosum_ventricles']
    corpus_vent.loc['total_cc_vol',:] = sum(corpus_vent.loc['cc_anterior':'cc_posterior','corpus_collosum_ventricles'].tolist())

    cv = corpus_vent.reset_index()

    #Volumne percentage data
    tr_dat = pd.read_csv('data/{}.csv'.format(mri))

    tr_dat.columns = list(pd.DataFrame(tr_dat.columns).iloc[:,0].map(lambda r: r.lower()))

    traindat = tr_dat[left_features + right_features]
    tr_min = {}
    tr_max = {}
    for col in left_features + right_features :
        tr_min[col] = traindat[col].min()
        tr_max[col] = traindat[col].max()

    for ind in dd['index'] :
        if ind in ['thalamus-proper', 'caudate','putamen','pallidum', 'hippocampus','amygdala', 'accumbens-area', 'cerebellum-cortex','cerebellum-white-matter','lateral-ventricle','inf-lat-vent'] :
            ind_lh = 'left-' + ind
            ind_rh = 'right-' + ind

        elif ind not in ['thalamus-proper', 'caudate','putamen','pallidum', 'hippocampus','amygdala', 'accumbens-area', 'cerebellum-cortex','cerebellum-white-matter','lateral-ventricle','inf-lat-vent'] and ind.split('_')[0] != 'total':
            ind_lh = 'lh_' + ind
            ind_rh = 'rh_' + ind

        else :
            pass

        d.loc[ind, 'left_per'] = normalization(pd.DataFrame([tr_min[ind_lh],tr_max[ind_lh],d.loc[ind, 'left']]), 'minmax').flatten()[-1]
        d.loc[ind, 'right_per'] = normalization(pd.DataFrame([tr_min[ind_rh],tr_max[ind_rh],d.loc[ind, 'right']]), 'minmax').flatten()[-1]
        d.loc[ind, 'total_per'] = normalization(pd.DataFrame([tr_min[ind_lh] + tr_min[ind_rh],tr_max[ind_lh] + tr_max[ind_rh],d.loc[ind, 'total']]), 'minmax').flatten()[-1]

    traindat_cc = tr_dat[cc_vent]
    tr_min_cc = {}
    tr_max_cc = {}
    for col in cc_vent :
        tr_min_cc[col] = traindat_cc[col].min()
        tr_max_cc[col] = traindat_cc[col].max()

    for ind in cv['index'] :
        if ind.split('_')[0] != 'total' :
            corpus_vent.loc[ind, 'CCV_per'] = normalization(pd.DataFrame([tr_min_cc[ind],tr_max_cc[ind],corpus_vent.loc[ind, 'corpus_collosum_ventricles']]), 'minmax').flatten()[-1]
        else :
            corpus_vent.loc[ind, 'CCV_per'] = normalization(pd.DataFrame([sum(list(tr_min_cc.values())[0:5]), sum(list(tr_max_cc.values())[0:5]), corpus_vent.loc[ind, 'corpus_collosum_ventricles']]), 'minmax').flatten()[-1]

    dat= pd.concat([d, corpus_vent], axis = 1).reset_index()
    dat.to_csv('{}_report.csv'.format(inputdir))
    print('Successfully generated !!!')
