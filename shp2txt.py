import geopandas as gpd
import json
import shapely
from shapely.geometry import Point, Polygon
import pandas as pd
import os

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

    # 使用pyproj将经纬度转换
def transform_points(points):
    return [gcj02_to_wgs84(lng, lat) for lng, lat in points]

    # 将link_coors字段的经纬度转换为坐标
def convert_coords(link_coors):
    coords = link_coors.split(";")
    return [tuple(map(float, coord.split(','))) for coord in coords]

    # 判断每个点是否在Polygon内
def is_in_boundary(coords, boundaries):
    for boundary in boundaries:
        if all(Point(coord).within(boundary) for coord in coords):
            return True
    return False

from shapely.geometry import LineString

# 创建 LineString 函数，将每一行的坐标转换为 LineString
def create_linestring(coords):
    return LineString(coords)

def main():
    CSV_file_path = '../GT_csv/GT_GaoDe.csv'
    df_GT = pd.read_csv(CSV_file_path)

    df_GT['dt_code'] = df_GT['dt_code'].astype(str)

    for district in ['130634', '411523', '130530', '130631', '469001', '130636', '130529', '341203', '130532', '130531', '130434', '620102', '469024', '130925', '360781', '610426', '130924', '131124', '620802', '130425', '130533', '131128', '411625', '130125', '520328', '130522', '140829', '410328', '341322', '130927', '610826', '451022', '610430']:# ['130434']:#['610700']:
    # ['500241', '610329', '141034', '341203', '360724', '360781', '522635', '341523', '130636', '360830', '520328', '520327', '610826', '450123', '411324', '130425', '430822', '411625', '130925', '610426', '532324', '130434', '540229', '411523', '360321', '130924', '530827', '410327', '522327', '130522', '622901', '431228', '530829', '630203', '431028', '621125', '430821', '131123', '130125', '513437', '620523', '610929', '433123', '410927', '140723', '140931', '130708', '140427', '130927', '622923', '522326', '610926', '141126', '130727', '451022', '131128', '610430', '469024', '130529', '532325', '620102', '360821', '530624', '520424', '469030', '610222', '410225', '530923', '513322', '533301', '620802', '410328', '610729', '431225', '430225', '341322', '130731', '130533', '130126', '520326', '340828', '511529', '522729', '140927', '522632', '360726', '510824', '451121', '433130', '530521', '140223', '522325', '130630', '530629', '610727', '620821', '511602', '450329', '140929', '610831', '532925', '540502', '520624', '630225', '130531', '469001', '610328', '140224', '530924', '520425', '532932', '610527', '511381', '620525', '140221', '522623', '513435', '420529', '140429', '141030', '431027', '131124', '520324', '340826', '431230', '130624', '140928', '422827', '532523', '131122', '620122', '450125', '411422', '532931', '520403', '130129', '410325', '130631', '520525', '520203', '130728', '430529', '340827', '610924', '533122', '141028', '522634', '411723', '140829', '451027', '130530', '361125', '610722', '141127', '360828', '422823', '640402', '532601', '130709', '522628', '610927', '141129', '540104', '411321', '520628', '513338', '140215', '140425', '510525', '540123', '130532', '130634']:

        if not os.path.exists('../GraphSamplingToolkit-main_improve_GE/'+str(district)+'/xyx_0'+'/groundtruth/'):
            os.makedirs('../GraphSamplingToolkit-main_improve_GE/'+str(district)+'/xyx_0'+'/groundtruth/')
        
        dt_code_city = str(district)[:-2]+str(0)+str(0)
        print(dt_code_city)

        df_filtered = df_GT[df_GT['dt_code']==dt_code_city].reset_index(drop=True)
        print(df_filtered)

        df_filtered['converted_coords'] = df_filtered['link_coors'].apply(convert_coords)

        df_filtered['transformed_coords'] = df_filtered['converted_coords'].apply(transform_points)

        # 加载GeoJSON文件
        geojson_boundary = gpd.read_file('../geojson/'+district+'.geojson')#

        # 创建Polygon边界
        boundaries = [Polygon(feature["geometry"]["coordinates"][0]) for feature in geojson_boundary.__geo_interface__["features"]]

        df_filtered['within_boundary'] = df_filtered['transformed_coords'].apply(lambda x: is_in_boundary(x, boundaries))
        print(df_filtered)
        # 过滤在边界内的点

        # 3. 将 `converted_coords` 中的点坐标转换为 LineString
        df_filtered['geometry'] = df_filtered[df_filtered['within_boundary'] == True]['transformed_coords'].apply(create_linestring)

        # 4. 构建 GeoDataFrame，并将 LineString 作为几何列
        gdf = gpd.GeoDataFrame(df_filtered, geometry='geometry')

        # 5. 设置坐标参考系，假设原始数据使用的是 WGS84 (EPSG:4326)
        gdf.set_crs(epsg=4326, inplace=True)
        # gdf = gpd.GeoDataFrame(df_filtered[df_filtered['within_boundary'] == True], 
        #                     geometry=gpd.points_from_xy(df_filtered['converted_coords'].apply(lambda x: x[0][1]), df_filtered['converted_coords'].apply(lambda x: x[0][0])))
        # 
        # 设置坐标系
        # gdf.set_crs(epsg=4326, inplace=True)
        # print(gdf.columns)
        # print(gdf)

        polyline_gdf = gdf[gdf['geometry'].geom_type == 'LineString']
        # print(polyline_gdf)


        # 写入点坐标和编号的文本文件
        with open('../GraphSamplingToolkit-main_improve_GE/'+str(district)+'/xyx_0'+'/groundtruth/'+str(district)+'_groundtruth_txt_vertices_osm.txt', 'w') as f_points:
            point_dict = {}  # 用于存储点的经纬度和对应的编号
            point_id = 1
            for index, row in polyline_gdf.iterrows():
                polyline_coords = list(row['geometry'].coords)
                # f_points.write(f"Polyline {index + 1} coordinates:\n")
                for coord in polyline_coords:
                    point = [coord[0],coord[1]]
                    # 将边界列表转换为多边形对象
                    # 将点转换为点对象
                    point_obj = Point(point)
                    # print(point_obj)
                    # # 判断点是否在多边形内
                    # is_within = point_obj.within(polygon)
                    # print(is_within)
                    # if is_within == False:
                    #     continue

                    coord_str = f"{coord[0]},{coord[1]}"  # 将经纬度转换为字符串
                    if coord_str not in point_dict:
                        point_dict[coord_str] = point_id
                        f_points.write(f"{point_id},{coord[0]},{coord[1]}\n")
                        point_id += 1

        # 判断点之间是否相连，并写入连接关系的文本文件
        with open('../GraphSamplingToolkit-main_improve_GE/'+str(district)+'/xyx_0'+'/groundtruth/'+str(district)+'_groundtruth_txt_edges_osm.txt', 'w') as f_connections:
            line_number = 1  # 行号
            for index, row in polyline_gdf.iterrows():
                polyline_coords = list(row['geometry'].coords)
                for i in range(len(polyline_coords) - 1):
                    coord1_str = f"{polyline_coords[i][0]},{polyline_coords[i][1]}"
                    coord2_str = f"{polyline_coords[i + 1][0]},{polyline_coords[i + 1][1]}"
                    if coord1_str in point_dict and coord2_str in point_dict:
                        point_id1 = point_dict[coord1_str]
                        point_id2 = point_dict[coord2_str]
                        f_connections.write(f"{line_number},{point_id1},{point_id2},1\n")
                        line_number += 1


if __name__ == "__main__":
    main()