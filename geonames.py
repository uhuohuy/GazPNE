# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:00:59 2019
it is used to extract name of public buildings from geonames, which are not provided by OSMNames 
@author: hu_xk
"""
import re
import matplotlib.pyplot as plt
import pandas as pd
import shapefile
import shapely
from utility import write_place 
import json
import geopandas as gpd
import numpy as np
import geojson
from operator import itemgetter 
from collections import Counter
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
def save_geo_entities(file,data):
    #save file
    w_truth = shapefile.Writer(file)
    w_truth.field('name', 'C');
    for i, lat in enumerate(data['latitude'].values):
        w_truth.point(data['longitude'].values[i],lat);
        w_truth.record(data['asciiname'].values[i]);
    w_truth.close()

country = 'IN'
#Entity features http://www.geonames.org/export/codes.html
MEDICAL_FEATURES = ['HSP','HSPC','HSPD','HSPL','CTRM']
WATER_FEATURES = ['PMPW','TNKD','MLWTR','LK','LKC','LKI','LKO']
EDUCATION_FEATURES = ['SCH','SCHA','SCHL','SCHM','SCHN','SCHT','UNIV','UNIP']
LIBRARY_FEATURES = ['LIBR']
POWER_FEATURES = ['PS','PSH','PSN']
POLICE_FEATURES = ['PP']
CLIFF_FEATURES = ['cliff']
AIR_FEATURES = ['AIRF','AIRH','AIRP','AIRB']
TRIBAL_FEATURES = ['TRB']
CHURCH_FEATURES = ['CH']
PARK_FEATURES = ['AMUS','CMN','PRK']
POST_FEATURES = ['PO','PSTP']
MARKET_FEATURES = ['MKT']
HOTEL_FEATURES = ['HTL']
THEATER_FEATURES = ['THTR','AMTH','OPRA']
GARDEN_FEATURES = ['GDN','ZOO']
POPULAR_FEATURES = ['PPL']
BANK_FEATURES = ['BANK']

geo_file = 'data/'+country + '.txt'
feature_names = ['geonameid','name','asciiname','alternatenames','latitude','longitude','feature_class','feature_code','country_code','cc2','admin1','admin2','admin3','admin4','population','elevation','dem','timezone','modi_date']
result = pd.read_csv(geo_file,sep='\t', names = feature_names);
names = result['name']
total_file = 'data/'+country.lower()+'_total_geonames.txt'
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
popular = result[result['feature_code'].isin(POPULAR_FEATURES)]  
bank = result[result['feature_code'].isin(BANK_FEATURES)]  

total_result = []
osm_result = open(osm_result_file,'w')
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



pure_result = extract_place(result)

write_place(osm_result_file,total_result)
write_place(total_file, pure_result)






