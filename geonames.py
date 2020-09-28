# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:00:59 2019
it is used to extract name of public buildings from geonames, which are not provided by OSMNames 
"""
import re
import pandas as pd
from utility import write_place 
import argparse
def extract_place(pd_item):
    return_places = []
    for item in pd_item['name']:
        if '-' not in item and '/' not in item and '(' not in item:
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", item)
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            new_corpus = []
            for cor in corpus:
            	all_ascii = ''.join(char for char in cor if ord(char) < 127)
            	new_corpus.append(all_ascii)
            corpus = [x for x in new_corpus if x and (len(x) < 2 or (len(x)>=2 and not (x[0]== '(' and x[len(x)-1]== ')')))]
            return_places.append(tuple(corpus))
    return return_places


def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--c', type=str, default='IN')

    args = parser.parse_args()
    print ('country: '+str(args.c))
    country = args.c #'US'
    #Entity features http://www.geonames.org/export/codes.html
    MEDICAL_FEATURES = ['HSP','HSPC','HSPD','HSPL','CTRM']
    EDUCATION_FEATURES = ['SCH','SCHA','SCHL','SCHM','SCHN','SCHT','UNIV','UNIP']
    LIBRARY_FEATURES = ['LIBR']
    AIR_FEATURES = ['AIRF','AIRH','AIRP','AIRB']
    CHURCH_FEATURES = ['CH']
    PARK_FEATURES = ['AMUS','CMN','PRK']
    POST_FEATURES = ['PO','PSTP']
    MARKET_FEATURES = ['MKT']
    HOTEL_FEATURES = ['HTL']
    THEATER_FEATURES = ['THTR','AMTH','OPRA']
    GARDEN_FEATURES = ['GDN','ZOO']
    BANK_FEATURES = ['BANK']
    
    geo_file = 'data/'+country + '.txt'
    feature_names = ['geonameid','name','asciiname','alternatenames','latitude','longitude','feature_class','feature_code','country_code','cc2','admin1','admin2','admin3','admin4','population','elevation','dem','timezone','modi_date']
    result = pd.read_csv(geo_file,sep='\t', names = feature_names);
    osm_result_file = 'data/'+country.lower()+'_geonames.txt'
    
    hospitals = result[result['feature_code'].isin(MEDICAL_FEATURES)]
    schools = result[result['feature_code'].isin(EDUCATION_FEATURES)]
    airs = result[result['feature_code'].isin(AIR_FEATURES)]
    churchs = result[result['feature_code'].isin(CHURCH_FEATURES)]
    library = result[result['feature_code'].isin(LIBRARY_FEATURES)]
    post = result[result['feature_code'].isin(POST_FEATURES)]
    park = result[result['feature_code'].isin(PARK_FEATURES)]
    market = result[result['feature_code'].isin(MARKET_FEATURES)]  
    hotel = result[result['feature_code'].isin(HOTEL_FEATURES)]  
    theater = result[result['feature_code'].isin(THEATER_FEATURES)]  
    garden = result[result['feature_code'].isin(GARDEN_FEATURES)]  
    bank = result[result['feature_code'].isin(BANK_FEATURES)]  
    
    total_result = []
    total_result.extend(extract_place(hospitals))
    total_result.extend(extract_place(schools))
    total_result.extend(extract_place(airs))
    total_result.extend(extract_place(churchs))
    total_result.extend(extract_place(library))
    total_result.extend(extract_place(post))
    total_result.extend(extract_place(park))
    total_result.extend(extract_place(market))
    total_result.extend(extract_place(hotel))
    total_result.extend(extract_place(theater))
    total_result.extend(extract_place(garden))
    total_result.extend(extract_place(bank))
    
    write_place(osm_result_file,total_result)
if __name__ == '__main__':
    main()
    
    




