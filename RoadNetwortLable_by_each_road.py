import os
from PIL import Image
import numpy as np
import geopandas as gpd
import PIL.Image
import cv2
PIL.Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math
import scipy.io as scio
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import math
#import geopandas
#import osmnx as ox
import urllib
import json
import argparse
from shapely.geometry import Point, Polygon
import json

def is_point_in_polygon(geojson_data, point):
    # 从GeoJSON提取多边形的坐标
    # polygon_coords = geojson_data['features'][0]['geometry']['coordinates'][0]
    
    # 创建Shapely的Polygon对象
    polygon = geojson_data#Polygon(polygon_coords)
    
    # 创建Shapely的Point对象
    point = Point(point)
    
    # 检查点是否在多边形内部
    return polygon.contains(point)


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


class Geocoding:
    def __init__(self, api_key):
        self.api_key = api_key

    def geocode(self, address):
        """
        利用高德geocoding服务解析地址获取位置坐标
        :param address:需要解析的地址
        :return:
        """
        geocoding = {'s': 'rsv3',
                     'key': self.api_key,
                     'city': '全国',
                     'address': address}
        geocoding = urllib.urlencode(geocoding)
        ret = urllib.urlopen("%s?%s" % ("http://restapi.amap.com/v3/geocode/geo", geocoding))

        if ret.getcode() == 200:
            res = ret.read()
            json_obj = json.loads(res)
            if json_obj['status'] == '1' and int(json_obj['count']) >= 1:
                geocodes = json_obj['geocodes'][0]
                lng = float(geocodes.get('location').split(',')[0])
                lat = float(geocodes.get('location').split(',')[1])
                return [lng, lat]
            else:
                return None
        else:
            return None


def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def point2geo0(p1, p2, im_height, im_width, para0, para1, para3, para5):
    delta_x = int((p1 - para0)/para1)
    delta_y = int((p2 - para3)/para5)

    if delta_y <im_height and delta_x<im_width:
            # if np.sum(im_data[:,delta_y,delta_x])<0:
                    # break
        return delta_y
    else:
        return -1
        
def point2geo1(p1, p2, im_height, im_width, para0, para1, para3, para5):
    delta_x = int((p1 - para0)/para1)
    delta_y = int((p2 - para3)/para5)

    if delta_y <im_height and delta_x<im_width:
        # if np.sum(im_data[:,delta_y,delta_x])<0:
                # break
        return delta_x
    else:
        return -1

def RoadNetwortLable_by_each_road(year,district):

    if os.path.exists('../temp_output/'+district+'_GT_primary_'+str(year)+'.png'):
        return 0

    if not os.path.exists('../temp_output/'+district+'_road_label_by_image_'+str(year)):
        os.makedirs('../temp_output/'+district+'_road_label_by_image_'+str(year))

    if not os.path.exists('../temp_output/'+district+'_width3_'+str(year)):
        os.makedirs('../temp_output/'+district+'_width3_'+str(year))

    # df = pd.read_csv('../data/district_boundary_long_lat3.csv')
    # district_cn = list(df[df['latin']==district]['district'])[0]
    # dt_code_cn = list(df[df['latin']==district]['dt_code'])[0]
    # pd_dict_GT = pd.read_csv('../data/tilefile_zl16_20_plus_20/'+ district_cn+'.csv')
    # img_name_list_all_GT = []
    # for i in range(len(pd_dict_GT)):
    #     img_name_list_all_GT.append(str(pd_dict_GT.at[i,'y_tile'])+'_'+str(pd_dict_GT.at[i,'x_tile']))

    # xiaoxian_road_GT_file = 'driving_road_xiaoxian_0_2020.geojson'
    # osm_road = gpd.read_file(xiaoxian_road_GT_file)

    district_road_GT_file = '../GT_csv/GT_GaoDe.csv' ############
    osm_road_full = pd.read_csv(district_road_GT_file)
    osm_road_full['dt_code'] = osm_road_full['dt_code'].astype(str)
    dt_code_city = str(district)[:-2]+str(0)+str(0)
    print(dt_code_city)
    osm_road = osm_road_full[(osm_road_full['dt_code']==dt_code_city) & (osm_road_full['year']==year)] ########最后两位置为00
    osm_road.reset_index(inplace=True)
    print(len(osm_road))
    print(osm_road)
#######将县和年份对应的路网提取
    img_name_list_all_GT = []
    for road_idx in range(len(osm_road)): ###################original geojson
        geo1 = osm_road.at[road_idx,'link_coors']  #每一条路先制成一个label， geometry为一个lon,lat list
        #####geo1 此时为一个字符串list 类似于 116.396574,34.639533;116.396547,34.639399;116.396676,34.638675;116.396703,34.638433
        road_name = osm_road.at[road_idx,'id'] ##希望每条路可以有一个标号,类似于一个unique id,如果没有,我再修改一下.. 
        # if road_name!=225534870:
        #     continue
        print(district,road_name)
        
        # geo1= list(geo1.geoms)
        # point_list = []
        p1_list = []
        p2_list = []
        geojson_data = gpd.read_file('../geojson/'+str(district)+'.geojson')
        for g in list(geo1.split(';')):
            lng_gcj = float(g.split(',')[0])
            lat_gcj = float(g.split(',')[1])
            lng_wgs, lat_wgs = gcj02_to_wgs84(lng_gcj,lat_gcj)
            # 需要检查的点（经度, 纬度）
            point_to_check = Point(lng_wgs, lat_wgs)

            # 检查点是否在多边形内部
            is_inside = geojson_data.contains(point_to_check).any()
            # print(is_inside)
            if int(is_inside):
                p1_list.append(lng_wgs)
                p2_list.append(lat_wgs)
        if len(p1_list) == 0:
            continue

            # point_list.append(g)

        # print(len(point_list))
        # print(point_list[:10])

        # p1_list = [float(x.split(',')[0]) for x in point_list]
        # p2_list = [float(x.split(',')[1]) for x in point_list]

        min_p1 = min(p1_list)
        max_p1 = max(p1_list)
        min_p2 = min(p2_list)
        max_p2 = max(p2_list)
        # print(min_p2, max_p2, min_p1, max_p1)

        min_x_tmp,min_y_tmp = deg2num(max_p2,min_p1,16)
        max_x_tmp,max_y_tmp = deg2num(min_p2,max_p1,16)

        min_x = min_x_tmp
        min_y = min_y_tmp
        
        max_x = max_x_tmp+1
        max_y = max_y_tmp+1
        # print(min_y, max_y, min_x, max_x)

        top_left_lat, top_left_lng = num2deg(min_x, min_y, 16)
        bottom_right_lat, bottom_right_lng = num2deg(max_x+1, max_y+1, 16)

        mask_width_now = max_x-min_x+1
        mask_height_now = max_y-min_y+1

        mask_width = mask_width_now
        mask_height = mask_height_now
        # print(mask_height, mask_width)

        delta_lng = (bottom_right_lng - top_left_lng)/mask_width/256
        delta_lat = (bottom_right_lat - top_left_lat)/mask_height/256

        pd_dict = pd.DataFrame({'p1': p1_list, 'p2': p2_list})
        pd_dict.drop_duplicates(subset = ['p1','p2'], inplace = True)
        # pd_dict.to_csv('ss/point_file.csv', index = False)

        # 0,3 左上角位置
        # 1 像元宽度
        # 5 像元高度
        para = [top_left_lng,delta_lng,0, top_left_lat,0,delta_lat]
        im_height = mask_height*256
        im_width = mask_width*256

        pd_dict['row'] = 0
        pd_dict['col'] = 0

        pd_dict['row'] = pd_dict.apply(lambda x: point2geo0(x['p1'], x['p2'], \
                    im_height, im_width, para[0], para[1], para[3], para[5]), axis = 1)
        pd_dict['col'] = pd_dict.apply(lambda x: point2geo1(x['p1'], x['p2'], \
                    im_height, im_width, para[0], para[1], para[3], para[5]), axis = 1)
        pd_dict.drop_duplicates(subset = ['row','col'], inplace = True)
        # print('len_pd_dict', len(pd_dict))
        pd_dict_new = pd_dict.drop(pd_dict[(pd_dict['row']<0) | (pd_dict['col']<0)|
                                        (pd_dict['row']>=im_height) | (pd_dict['col']>=im_width)].index)

        label_array = np.zeros((im_height, im_width))

        pd_dict_new = pd_dict_new.reset_index(drop = True)

        for k in range(len(pd_dict_new)-1):
            # print((pd_dict_new.at[k,'row'],pd_dict_new.at[k,'col']), (pd_dict_new.at[k+1,'row'],pd_dict_new.at[k+1,'col']))
            label_array = cv2.line(label_array, (pd_dict_new.at[k,'col'],pd_dict_new.at[k,'row']), (pd_dict_new.at[k+1,'col'], \
                                                                                                    pd_dict_new.at[k+1,'row']), (255, 255, 0), 1) #start_point, end_point, color, thickness


        #####################OSM label line color image

        for i in range(mask_height):
            for j in range(mask_width):
                img_name = str(i+min_y)+'_'+str(j+min_x)
                img_name_list_all_GT.append(img_name)

                # if img_name not in img_name_list_all_GT:
                #     continue

                if np.sum(label_array[(i)*256:(i+1)*256, (j)*256:(j+1)*256])==0:
                    continue

                im = Image.fromarray((label_array[(i)*256:(i+1)*256, (j)*256:(j+1)*256]))
                if not os.path.exists('../temp_output/'+district+'_width3_'+str(year)+'/'):
                    os.makedirs('../temp_output/'+district+'_width3_'+str(year)+'/')
                im.convert('L').save('../temp_output/'+district+'_width3_'+str(year)+'/'+district+ '_' + str(year) + '_OSM_road_label_'+str(road_name)+'_'+img_name+'.png')


    ##############################将同属于一张图像的所有路网综合起来
    # road_list_one_district_all = glob.glob('../temp_output/'+district+'_width3_'+str(year)+'/*.png')
    # road_list_one_district_all_sat_name = [x.split('/')[-1].split('_')  for x in road_list_one_district_all ] #'+district+ '_' + str(year) + '_OSM_road_label_'+str(road_name)+'_'+img_name+'
    pd_dict = pd.DataFrame({'img_name':img_name_list_all_GT})
    pd_dict['Top_Edge_Latitude'] = 0.0
    pd_dict['Bottom_Edge_Latitude'] = 0.0
    pd_dict['Right_Edge_Longitude'] = 0.0
    pd_dict['Left_Edge_Longitude'] = 0.0
    pd_dict['y_tile'] = 0
    pd_dict['x_tile'] = 0

    for pd_idx in range(len(pd_dict)):

        y_tile_pd = int(pd_dict.at[pd_idx,'img_name'].split('_')[0])
        x_tile_pd = int(pd_dict.at[pd_idx,'img_name'].split('_')[1])
        pd_dict.at[pd_idx,'y_tile'] = y_tile_pd
        pd_dict.at[pd_idx,'x_tile'] = x_tile_pd 
        (pd_dict.at[pd_idx,'Top_Edge_Latitude'],pd_dict.at[pd_idx,'Left_Edge_Longitude']) = num2deg(x_tile_pd, y_tile_pd, 16)
        (pd_dict.at[pd_idx,'Bottom_Edge_Latitude'],pd_dict.at[pd_idx,'Right_Edge_Longitude']) = num2deg(x_tile_pd+1, y_tile_pd+1, 16)

    # if not os.path.exists('../data/tilefile_zl16_20_plus_20/'):
    #     os.makedirs('../data/tilefile_zl16_20_plus_20/')
    # pd_dict = pd.read_csv('../data/tilefile_zl16_20_plus_20/'+ district+'.csv')
    # pd.read_csv(district+'.csv')
    if not os.path.exists('../data/tilefile_zl16_20_plus_20/'):
        os.makedirs('../data/tilefile_zl16_20_plus_20/')
    pd_dict.to_csv('../data/tilefile_zl16_20_plus_20/'+ district+'.csv', index=False)

    label_list = glob.glob('../temp_output/'+district+'_width3_'+str(year)+'/*.png')
    # print(label_list[:5])

    label_img_list = [x.split('.')[-2].split('_')[-2]+'_'+x.split('.')[-2].split('_')[-1] for x in label_list]

    print(label_img_list[:5])

    for k in range(len(pd_dict)):
        # y_tile = pd_dict.at[k,'y_tile']
        # x_tile = pd_dict.at[k,'x_tile']
        # img_name = str(y_tile)+'_'+str(x_tile)
        img_name = pd_dict.at[k,'img_name']
        # print(img_name)
        # img_name = '26174_54068'

        b = []
        for index, nums in enumerate(label_img_list):
            if nums == str(img_name):
                b.append(index)
        # print(b)

        if len(b)==0:
            continue

        if len(b) > 1:
            label_img = np.array(Image.open(label_list[b[0]]))
            for j in b[1:]:
                label_img_tmp = np.array(Image.open(label_list[j]))
                label_img = label_img+label_img_tmp
            label_img[label_img>0] = 255
        else:
            label_img = np.array(Image.open(label_list[b[0]]))
            label_img[label_img>0] = 255

        im = Image.fromarray(label_img)            
        if not os.path.exists('../temp_output/'+district+'_road_label_by_image_'+str(year)+'/'):
            os.makedirs('../temp_output/'+district+'_road_label_by_image_'+str(year)+'/')
        im.convert('L').save('../temp_output/'+district+'_road_label_by_image_'+str(year)+'/'+img_name+'.png')  #xiaoxian
