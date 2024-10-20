import pandas as pd
import glob
from concat_all_label_image import *
from RoadNetwortLable_by_each_road import *
from convert_matlab_code import *
from graph_extraction_pixel import *
from main_pixel import *
from shp2txt_transform_new import *
from simplify_road import *
from transform_coord_grid import *
from create_shp_from_txt_json import *
from filter_road_by_bound import *
import mapcompare_GE_0001
import sys
import mapcompare_GE
import average_statistics

if __name__ == '__main__':

    year = 2020
    district_list = ['130634', '411523', '610426','130631','130636', '130529', '341203', '130532', '130531', '130434']#, '130924' ,'131124',  '130925', '360781'#,'469024', '130530',  '131124', '620802', '130425', '130533', '131128', '411625', '130125', '520328', '130522', '140829', '410328', '341322', '130927', '610826', '451022', '610430']
    for district in district_list:
        if os.path.exists('results_pixel_coored_final_shp/'+district+'_'+str(year)+'-GE-18-final-wgs_inbound.shp'):
            continue
        RoadNetwortLable_by_each_road(year,district)
        concat_all_label_image(year,district)
        convert_matlab_code(district,year)
        main_pixel(district,year)
        simplify_road(district,year)
        transform_coord_grid(district, year)
        create_shp_from_txt_json(district, year)
        filter_road_by_bound(district,year)
        shp2txt_transform_new(district, year)
        
        county = district
        del_list = os.listdir('../temp_output/'+county+'_road_label_by_image_'+str(year)+'/')
        #'../temp_output/'+district+'_road_label_by_image_'+str(year)+'/'
        for f in del_list:
            file_path = os.path.join('../temp_output/'+county+'_road_label_by_image_'+str(year)+'/', f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        del_list = os.listdir('../temp_output/'+county+'_width3_'+str(year)+'/')
        #'../temp_output/'+district+'_width3_'+str(year)+'/'
        for f in del_list:
            file_path = os.path.join('../temp_output/'+county+'_width3_'+str(year)+'/', f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.removedirs('../temp_output/'+county+'_road_label_by_image_'+str(year))
        os.removedirs('../temp_output/'+county+'_width3_'+str(year))

    mapcompare_GE.main()
    # mapcompare_GE_0001.main()
    average_statistics.main()