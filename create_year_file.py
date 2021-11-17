import pandas as pd
import pycountry

BACI_FORMAT = 'BACI/BACI_HS92_Y{}_V202102.csv'
WORLD_BANK_FILE = 'world_bank_gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3263806.csv'
OUTPUT_EDGE_INDEX = 'output/X_{}.csv'
OUTPUT_NODE_FEATURES = 'output/Y_{}.csv'

def create_files(year, k=15):
    baci = pd.read_csv(BACI_FORMAT.format(year))
    baci = baci.groupby(['i','j']).sum() # sum together all categories of objects 
    baci = baci.sort_values(['v'], ascending=False).groupby(['i']).head(k).reset_index().filter(['i','j']) # keep only top k edges by export value

    def convert_row(row):
        country = pycountry.countries.get(alpha_3=row.Country_Code)
        if country is None:
            return -1
        else:
            return int(country.numeric)

    wb = pd.read_csv(WORLD_BANK_FILE)
    wb['iso_code'] = wb.apply(convert_row, axis=1)
    wb = wb[wb['iso_code'].isin(baci['i'])].filter(['iso_code', str(year+1)]) # keep rows corresponding to countries that are in the BACI dataset
    wb = wb[wb[str(year+1)] > 0] # some rows have zero GDP data, remove them
    baci = baci[baci['i'].isin(wb['iso_code']) & baci['j'].isin(wb['iso_code'])] # ensure that there are no edges that involve countries for which we lack GDP

    assert(wb.shape[0] == baci['i'].nunique()) # GDP and exporters should be of same cardinality
    baci.to_csv(OUTPUT_EDGE_INDEX.format(year), index=False)
    wb.to_csv(OUTPUT_NODE_FEATURES.format(year), index=False)

for year in range(1995, 2019):
    print(year)
    create_files(year)
