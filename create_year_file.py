import pandas as pd
import numpy as np
import pycountry
import warnings

warnings.filterwarnings(action='once') # using a deprecated median function that throws warning, to tired to fix it now
BACI_FORMAT = 'BACI/BACI_HS92_Y{}_V202102.csv'
WORLD_BANK_GDP_FILE = 'world_bank_gdp/gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3263806.csv'
WORLD_BANK_POP_FILE = 'world_bank_gdp/pop/API_SP.POP.TOTL_DS2_en_csv_v2_3358390.csv'
WORLD_BANK_CPI_FILE = 'world_bank_gdp/cpi/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_3358170.csv'
WORLD_BANK_EMP_FILE = 'world_bank_gdp/emp/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_3358447.csv'
INPUT_NODE_FEATURES = 'output/X_NODE_{}.csv'
OUTPUT_EDGE_INDEX = 'output/X_EDGE_{}.csv'
OUTPUT_NODE_TARGETS = 'output/Y_{}.csv'
NUM_PRODS = 10

feature_datasets = [WORLD_BANK_POP_FILE, WORLD_BANK_CPI_FILE, WORLD_BANK_EMP_FILE]

def create_files(year, k=15):
    baci = pd.read_csv(BACI_FORMAT.format(year))
    baci = baci.groupby(['i','j']).sum() # sum together all categories of objects 
    baci = baci.sort_values(['v'], ascending=False).groupby(['i']).head(k).reset_index().filter(['i','j']) # keep only top k edges by export value
    
    edge_baci = create_edge_features(year) # add edge features consisting of q values for top 10 products for each edge.
    baci = pd.merge(baci, edge_baci, how='left') #merge the features with the edges we've filtered by expoort value above.
    
    def convert_row(row):
        country = pycountry.countries.get(alpha_3=row['Country Code'])
        if country is None:
            return -1
        else:
            if country.alpha_3 =='USA':
                return 842    # for some reason the USA sometimes uses 840 and other times 842
            return int(country.numeric)

    # Pre-process and write GDP data
    wb_gdp = pd.read_csv(WORLD_BANK_GDP_FILE, header=2)
    wb_gdp['iso_code'] = wb_gdp.apply(convert_row, axis=1)
    wb_gdp = wb_gdp[wb_gdp['iso_code'].isin(baci['i'])].filter(['iso_code', str(year+1)]) # keep rows corresponding to countries that are in the BACI dataset
    wb_gdp = wb_gdp[wb_gdp[str(year+1)] > 0] # some rows have zero GDP data, remove them
    baci = baci[baci['i'].isin(wb_gdp['iso_code']) & baci['j'].isin(wb_gdp['iso_code'])] # ensure that there are no edges that involve countries for which we lack GDP
    assert(wb_gdp.shape[0] == baci['i'].nunique()) # GDP and exporters should be of same cardinality   
    
    # Pre-process and write population / CPI / unemployment data
    wb_feat = []
    for i, file in enumerate(feature_datasets):
        wb = pd.read_csv(file, header=2)
        
        # Impute any NaNs with median data of column if feature is cpi or unemployment (gdp probably wouldn't work)
        feature = file.split('/')[1]
        if feature == 'cpi' or feature == 'emp':
            with warnings.catch_warnings(record=True):
                wb = wb.fillna(wb.median())    # this will be deprecated at some point, but i'm not tryna fix it rn

        # Configure feature dataframe headers 
        wb['iso_code'] = wb.apply(convert_row, axis=1)
        if i == 0:
            wb = wb[wb['iso_code'].isin(baci['i'])].filter(['iso_code', str(year+1)])
            wb.columns = ['iso_code', feature]
        else:
            wb = wb[wb['iso_code'].isin(baci['i'])].filter([str(year+1)])
            wb.columns = [feature]
        
        # Ensure that all feature dataframes have the same length as the target gdp dataframe
        assert len(wb) == len(wb_gdp)
        wb_feat.append(wb)
        
    wb_feat = pd.concat(wb_feat, axis=1)
    wb_feat.to_csv(INPUT_NODE_FEATURES.format(year), index=False)       
    baci.to_csv(OUTPUT_EDGE_INDEX.format(year), index=False)

def create_edge_features(year):
    # baci = pd.read_csv(BACI_FORMAT.format(year))
    # baci = baci.groupby(['i','j'])
  
    baci = pd.read_csv(BACI_FORMAT.format(year))
    baci = baci.fillna(0) # some q values are missing, fill them with 0
    # baci = baci.groupby(['i','j'])
    top_products = baci.groupby(['k']).sum().sort_values(['v'],ascending=False).head(NUM_PRODS).reset_index()['k']
    k_to_idx = {id: i for (i, id) in enumerate(top_products)}
    
    # baci = baci.sort_values(['v'], ascending=False)
    feature_dict = {}
    baci = baci[baci['k'].isin(top_products)].filter(['i', 'j', 'k', 'q'])
    # print(baci.columns)
    
    d = {}
    def update_dict(i, j, k, q):
      i = int(i)
      j = int(j)
      if (i, j) not in d:
        d[(i,j)] = [0] * NUM_PRODS
      r = d[(i,j)] # get vec for a specific edge (i,j)
      r[k_to_idx[k]] = q # update vec for edge (i,j) and product k
      d[(i,j)] = r # rewrite the vec
        

    # baci['new'] = baci.apply(lambda r: update_dict(int(r['i']), int(r['j']), int(r['k']), r['q']), axis=1)
    baci.apply(lambda r: update_dict(int(r['i']), int(r['j']), int(r['k']), r['q']), axis=1)
    edge_features = np.vstack([d[(r['i'], r['j'])] for _, r in baci.iterrows()]) # create matrix of edge feature vecs we will write to the files.
    feature_names = ['f'+ str(i) for i in range(NUM_PRODS)] # name our edge features 'f0,...f9'
    baci[feature_names] = edge_features # write in the data to the baci dataframe
    return baci

if __name__ == "__main__":
    for year in range(1995, 2020):
        print(f"Processing year {year}", end='\r')
    create_files(year)
