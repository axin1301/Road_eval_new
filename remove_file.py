import os

year = 2020
district = '130434'

county = district
if os.path.exists('../temp_output/'+county+'_road_label_by_image_'+str(year)+'/'):
    del_list = os.listdir('../temp_output/'+county+'_road_label_by_image_'+str(year)+'/')
    #'../temp_output/'+district+'_road_label_by_image_'+str(year)+'/'
    for f in del_list:
        file_path = os.path.join('../temp_output/'+county+'_road_label_by_image_'+str(year)+'/', f)
        if os.path.isfile(file_path):
            os.remove(file_path)

if os.path.exists('../temp_output/'+county+'_width3_'+str(year)+'/'):
    del_list = os.listdir('../temp_output/'+county+'_width3_'+str(year)+'/')
    #'../temp_output/'+district+'_width3_'+str(year)+'/'
    for f in del_list:
        file_path = os.path.join('../temp_output/'+county+'_width3_'+str(year)+'/', f)
        if os.path.isfile(file_path):
            os.remove(file_path)

if os.path.exists('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list.txt'):
    os.remove('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list.txt')

if os.path.exists('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list_simplified.txt'):
    os.remove('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list_simplified.txt')

if os.path.exists('../temp_output/'+district+'_GT_primary_'+str(year)+'-17-bone.png'):
    os.remove('../temp_output/'+district+'_GT_primary_'+str(year)+'-17-bone.png')

if os.path.exists('../temp_output/'+district+'_GT_primary_'+str(year)+'.png'):
    os.remove('../temp_output/'+district+'_GT_primary_'+str(year)+'.png')

if os.path.exists('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list_simplified_edge.txt'):
    os.remove('results_pixel_bone_pred/results_pixel_bone_pred_'+district+'_'+str(year)+'_coord/coord_list_simplified_edge.txt')

if os.path.exists('results_pixel_bone_pred/pixel/'+district+'-'+str(17)+'-simplified.txt'):
    os.remove('results_pixel_bone_pred/pixel/'+district+'-'+str(17)+'-simplified.txt')