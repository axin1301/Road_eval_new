import sys
sys.path.append('../')
sys.path.append('../tptk/')
import argparse
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
# export CV_IO_MAX_IMAGE_PIXELS=1099511627776
import cv2
import pandas as pd
from tptk.common.mbr import MBR
from tptk.common.grid import Grid
from tptk.common.road_network import load_rn_shp
# from topology_construction.
from graph_extraction_pixel import GraphExtractor
from topology_construction.link_generation import LinkGenerator
from topology_construction.custom_map_matching import CustomMapMatching
from topology_construction.map_refinement import MapRefiner
import json
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from skimage import morphology,draw
import matplotlib.pyplot as plt

def get_test_mbr(conf):
    dataset_conf = conf['dataset']
    feature_extraction_conf = conf['feature_extraction']
    min_lat, min_lng, max_lat, max_lng = dataset_conf['min_lat'], dataset_conf['min_lng'], \
                                         dataset_conf['max_lat'], dataset_conf['max_lng']
    whole_region_mbr = MBR(min_lat, min_lng, max_lat, max_lng)
    whole_region_grid = Grid(whole_region_mbr, dataset_conf['nb_rows'], dataset_conf['nb_cols'])
    test_row_min, test_col_min, test_row_max, test_col_max = feature_extraction_conf['test_tile_row_min'], \
                                                             feature_extraction_conf['test_tile_col_min'], \
                                                             feature_extraction_conf['test_tile_row_max'], \
                                                             feature_extraction_conf['test_tile_col_max']
    tile_pixel_size = feature_extraction_conf['tile_pixel_size']
    test_row_min_idx = test_row_min * tile_pixel_size
    test_row_max_idx = test_row_max * tile_pixel_size
    test_col_min_idx = test_col_min * tile_pixel_size
    test_col_max_idx = test_col_max * tile_pixel_size

    test_region_lower_left_mbr = whole_region_grid.get_mbr_by_matrix_idx(test_row_max_idx, test_col_min_idx)
    test_region_min_lat, test_region_min_lng = test_region_lower_left_mbr.max_lat, test_region_lower_left_mbr.min_lng
    test_region_upper_right_mbr = whole_region_grid.get_mbr_by_matrix_idx(test_row_min_idx, test_col_max_idx)
    test_region_max_lat, test_region_max_lng = test_region_upper_right_mbr.max_lat, test_region_upper_right_mbr.min_lng
    test_region_mbr = MBR(test_region_min_lat, test_region_min_lng, test_region_max_lat, test_region_max_lng)
    return test_region_mbr

def main_pixel(district,year):
# if __name__ == '__main__':
    
#     # df = pd.read_csv('../../../../20_20_districts/district_boundary_long_lat.csv')

#     parser = argparse.ArgumentParser()
    # parser.add_argument('--phase', type=int, default=1, help='1,2,3,4')
    # parser.add_argument('--conf_path', help='the configuration file path')
    # parser.add_argument('--results_path', help='the path to the results directory')
    # parser.add_argument('--year')
    # parser.add_argument('--district')

    # opt = parser.parse_args()
    # district = opt.district
    # year = opt.year
    # district_list = ['130125', '130126', '130129', '130425', '130434', '130522', '130529', '130530', '130531', '130532', '130533', \
    #                     '130631', '130634', '130636', '130708', '130924', '130925', '130927', '131122', '131123', '131124', '131128', \
    #                         '140223', '140427', '140829', '140931', '141030', '141034', '340827', '341203']#['340827']#['141034']#['341203']
    # district_list = ['520603', '130522', '469024', '522636', '469029', '410225', '610328', '522630', '620102', '620102', '130533', '420921', '411423', '130529', '130532', '630203', '130531', '622924', '130631', '411424', '361125', '131123', '131122', '141034', '610430', '540524', '130925', '411422', '520602', '411327', '622927', '530626', '433124', '610428', '131128', '341203', '131124', '130434', '522635']
    # ['130631', '131123', '131122', '141034', '130925', '131128', '341203', '131124', '130434', '522635']#['610328', '130533', '130529', '130532', '130531']#['130522']#['341523', '130522', '532325', '530624', '610929', '430822', '530521', '411625', '410225', '130927', '360830', '469001', '511381', '620102', '360726', '130636', '530923', '610430', '340828', '410328', '140223', '140931', '430529', '469030', '621125', '520628', '610722', '532931', '130728', '450123', '620821', '511529', '451022', '130533', '540502', '430821', '141034', '140427', '130530', '361125', '610831', '130925', '610426', '520403', '622901', '533301', '630203', '131128', '620523', '610729', '520326', '522628', '513435', '430225', '360724', '522729', '360781', '340827', '450329', '411523', '640402', '522634', '140928', '540123', '431028', '141030', '532523', '510525', '130631', '360821', '130434', '532324', '630225', '530829', '520327', '410327', '522623', '610927', '451027', '540229', '411422', '431225', '522326', '410927', '513338', '513322', '450125', '140829', '130425', '140723', '500241', '433123', '520328', '431027', '141028', '610328', '530924', '620802', '510824', '410325', '141126', '131122', '620525', '530827', '130731', '610527', '130630', '530629', '130708', '431228', '522635', '522632', '411723', '411324', '610222', '610926', '140425', '130727', '451121', '141129', '130126', '532601', '520324', '140221', '520203', '141127', '130125', '140927', '610826', '140429', '140224', '522327', '130531', '130634', '520624', '130129', '620122', '540104', '341322', '360828', '140215', '131123', '130529', '522325', '422827', '360321', '431230', '622923', '130532', '532932', '140929', '610727', '610329', '610924', '520424', '433130', '532925', '130924', '520425', '469024', '130624', '513437', '422823', '411321', '420529', '340826', '341203', '511602', '130709', '520525', '131124', '533122']

    year_list = [2020]
    district_list = [district]
    ########################1741 started  1750 ended
    phase = 1
    # for year in [2017, 2021]:
    for year in year_list:#[2021]:
        for district in district_list:# ['jingyuxian']:#['lingqiuxian']:#['xixiangxian','shufuxian','guanghexian','danfengxian','jiangzixian','honghexian','liboxian','linquanxian',,'lingqiuxian']:
            print(year, district)

            if os.path.exists('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list_simplified_edge.txt'):
                break
            conf_path = 'test_rn_OSM_full_jingyuxian18.json' ##占位而已，不产生影响
            # results_path = '../data/tdrive_sample_thresh500/results_pred_'+district+'_'+str(year) + '/'
            # results_path = '../data/tdrive_sample_thresh500_improve/results_pred_'+district+'_'+str(year) + '/'
            # results_path = '../../../RoadNetworkValidation_new/RoadNetwork_Validation4OSMGT/data/tdrive_sample_improve2/results_pred_'+district+'_'+str(year) + '/'
            # results_path = './results_pixel_bone_pred_'+district+'_GE_'+str(year) + '/'#'_full_image/'
            results_path = './results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_GE_'+str(year) + '/'#'_full_image/'

            # conf_path = '../data/test_rn_pred_full_'+district+'.json' #'../../../RoadNetworkValidation_new/RoadNetworkValidation/DeepMG-master/data/test_rn_pred_full_'+district+'.json'
            # results_path = '../data/tdrive_sample/results_pred_'+district+'_'+str(year) + '/' #+'_dist5_'

            # print(opt)
            if not os.path.exists('../temp_output/'+district+'_GT_primary_'+str(year)+'-17-bone.png'):#district+'-17-bone.png'):
                continue
            # radius = 200
            with open(conf_path, 'r') as f:
                conf = json.load(f)
            # results_path = opt.results_path
            # traj_path = '../data/{}/traj/'.format(conf['dataset']['dataset_name'])
            extracted_rn_path = results_path + 'extracted_rn/'
            # linked_rn_path = results_path + 'linked_rn_'+str(radius)+'/'
            mm_on_linked_rn_path = results_path + 'mm_on_linked_rn/'
            final_rn_path = results_path + 'final_rn/'
            os.makedirs(mm_on_linked_rn_path, exist_ok=True)
            # topo_params = conf['topology_construction']
            # Graph Extraction
            # if opt.phase == 1:
            if phase == 1:
                mbr = get_test_mbr(conf)
                # skeleton = cv2.imread(results_path + 'pred_thinned.png', cv2.IMREAD_GRAYSCALE)
                # skeleton = cv2.imread(str(conf['dataset']['dataset_name'])+'.png', cv2.IMREAD_GRAYSCALE)
                # skeleton = Image.open(str(conf['dataset']['dataset_name'])+'_'+str(year)+'.png')

                # latin_name = str(conf['dataset']['dataset_name']).split('_')[-1]
                # cname = '灵丘县'#list(df[df['latin']==latin_name]['district'])[0]
                # print(cname+'  is being processed.')
                # skeleton = Image.open('../../../RoadNetworkValidation_new/RoadNetwork_Validation4OSMGT/temp_output/topology_construction_thresh500/2021improve2/pred_skeleton_'+cname+'_'+str(year)+'_2_improve.png') #这里无用
                # skeleton = Image.open('lingqiuxian_zl17_new_div_10.png') #
                # skeleton = Image.open('lingqiuxian-17-bone.png')
                # skeleton = Image.open('refined_pred_GE_jyx_2019_2_100_update.png')
                # skeleton = Image.open('pred_skeleton_靖宇县_2021_2_improve.png')
                skeleton = Image.open('../temp_output/'+district+'_GT_primary_'+str(year)+'-17-bone.png')#district+'-17-bone.png')
                # skeleton = Image.open('jingyuxian_zl17_new_div_10.png') #除以10的偏移很大
                # skeleton = Image.open('lingqiuxian_zl17_new_full_image.png') #这里无用
                # skeleton = Image.open('jingyuxian_zl17_new_full_image.png')
                skeleton = np.array(skeleton)
                road = skeleton
                # skeleton[skeleton<255] = 1 #############GT
                # skeleton[skeleton==255] = 0
                # skeleton[skeleton==1] = 255
                # plt.imshow(skeleton)
                # plt.show()
                # skeleton = skeleton.convert('L')
                # road = np.array(skeleton[:,:,0])
                print(road.shape)
                mbr.nb_rows = road.shape[0]
                mbr.nb_cols = road.shape[1]
                mbr.test_tile_row_max = road.shape[0]
                mbr.test_tile_col_max = road.shape[1]

                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                # road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
                # plt.imshow(road)
                # plt.show()

                road = np.uint8(road)
                road = Image.fromarray(road)

                road_seg = np.array(road)#[30000:40000,30000:40000]
                road_idx = np.where(road_seg > 0)

                bin_image = np.zeros((road_seg.shape[0],road_seg.shape[1]))
                bin_image[road_idx[0],road_idx[1]] = 1

                image = bin_image
                #实施骨架算法
                skeleton =morphology.skeletonize(image)
                skeleton = Image.fromarray(skeleton)
                skeleton = np.array(skeleton)
                # plt.imshow(skeleton)
                # plt.show()
                # print(np.array(road).shape)

                map_extractor = GraphExtractor(epsilon=10, min_road_dist=50,district = district,year = year)
                map_extractor.extract(skeleton, mbr, extracted_rn_path,district,year)
            # Link Generation
            # elif opt.phase == 2:
            # elif phase == 2:
            #     # link_generator = LinkGenerator(radius=topo_params['link_radius'])
            #     link_generator = LinkGenerator(radius=radius)
            #     # the extracted rn is undirected, while the linked rn is directed (bi-directional)
            #     extracted_rn = load_rn_shp(extracted_rn_path, is_directed=False)
            #     link_generator.generate(extracted_rn, linked_rn_path)
            # Custom Map Matching
            # elif opt.phase == 3:
            #     custom_map_matching = CustomMapMatching(linked_rn_path, topo_params['alpha'])
            #     filenames = os.listdir(traj_path)
            #     with Pool() as pool:
            #         pool.map(partial(custom_map_matching.execute,
            #                         traj_path=traj_path, mm_result_path=mm_on_linked_rn_path), filenames)
            # # Map Refinement
            # elif opt.phase == 4:
            #     linked_rn = load_rn_shp(linked_rn_path, is_directed=True)
            #     map_refiner = MapRefiner(topo_params['min_supp'])
            #     map_refiner.refine(linked_rn, mm_on_linked_rn_path, final_rn_path)
            else:
                raise Exception('invalid phase')
    
    # cname = '灵丘县'
    # year = 2021
    # k = os.path.exists('../pred_skeleton_灵丘县_2021_2_improve.png')
    # print(k)