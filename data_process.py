import sys
sys.path.append('build')
sys.path.append('tasks/R2R/fasterRCNN_w_vg')
import MatterSim
from os import listdir, makedirs
from os.path import join, exists
import numpy as np
import math
import pickle
import cv2
from tasks.R2R.fasterRCNN_w_vg.obj_detection import setup_obj_detection, _get_obj_detection
import pdb
import time

# Process data for 'Object Detection Result'
def read_data_category(data_path='/disk2/yunhokim/Matterport3DSimulator/data/'):
    scan_path = join(data_path, 'v1/scans')
    scan_types = listdir(scan_path)
    print('{} scan types to be processed'.format(len(scan_types)))

    viewpoint_types = dict()

    for scan_type in scan_types:
        final_path = join(join(scan_path, scan_type), 'matterport_skybox_images')
        imgs = listdir(final_path)
        viewpointIds = set()
        for img in imgs:
            viewpointId = img.split('_')[0]
            viewpointIds.add(viewpointId)  
        viewpointIds = list(viewpointIds)
        viewpoint_types[scan_type] = viewpointIds
        print('{} viewpoint types in "{}" scan type to be processed'.format(len(viewpointIds), scan_type))
    
    return scan_types, viewpoint_types

def setup_simulator(image_w=160, image_h=120, vfov=60):
    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(True)
    sim.setDiscretizedViewingAngles(True)
    sim.setCameraResolution(image_w, image_h)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDepthEnabled(True)
    sim.initialize()

    return sim

def make_save_path(scanId, viewpointId, viewIndex, data_type):
    assert data_type in ['pickle', 'img', 'npz_depth', 'npz_n_navigable']

    file_name = viewpointId + '_' + str(viewIndex)
    if data_type == 'pickle':
        file_name += '.pickle'
    elif data_type == 'img':
        file_name += '.png'
    elif data_type == 'npz_depth' or data_type == 'npz_n_navigable':
        file_name += '.npz'

    if data_type == 'pickle':
        folder_name = join('obj_detection_result', scanId)
        if not exists(folder_name):
            makedirs(folder_name)
        file_name = join(folder_name, file_name)
    elif data_type == 'img':
        folder_name = join('obj_detection_image', scanId)
        if not exists(folder_name):
            makedirs(folder_name)
        file_name = join(folder_name, file_name)
    elif data_type == 'npz_depth':
        folder_name = join('depth_result', scanId)
        if not exists(folder_name):
            makedirs(folder_name)
        file_name = viewpointId + '.npz'
        file_name = join(folder_name, file_name)
    elif data_type == 'npz_n_navigable':
        folder_name = join('navigable_result', scanId)
        if not exists(folder_name):
            makedirs(folder_name)
        file_name = viewpointId + '.npz'
        file_name = join(folder_name, file_name)

    return file_name

def get_obj_detection_datas(scan_types, viewpoint_types):
    # Setup simulator
    sim = setup_simulator()

    # Setup object detector
    fasterRCNN, args, classes, class_indexs, im_data, im_info, num_boxes, gt_boxes = setup_obj_detection()

    for scan_type in scan_types:
        print('Processing {}/{}'.format(scan_types.index(scan_type)+1, len(scan_types)))
        viewpoint_types_Ids = viewpoint_types[scan_type]
        for viewpoint_type in viewpoint_types_Ids:
            try:
                index_history = []
                sim.newEpisode([scan_type], [viewpoint_type], [0], [0])

                for i in range(12):
                    state = sim.getState()[0]
                    rgb = np.array(state.rgb, copy=False)
                    # cv2.imshow('Python RGB', rgb)
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    img_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='img')
                    total_detected_result = _get_obj_detection(rgb, img_path, fasterRCNN, args, classes, class_indexs,\
                                                               im_data, im_info, num_boxes, gt_boxes)

                    detected_result_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='pickle')
                    with open(detected_result_path, 'wb') as f:
                       pickle.dump(total_detected_result, f)
	            
                    sim.makeAction([0], [1], [0])
	        
                sim.makeAction([0], [0], [1])
                for i in range(12):
                    state = sim.getState()[0]
                    rgb = np.array(state.rgb, copy=False)
                    # cv2.imshow('Python RGB', rgb)
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    img_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='img')
                    total_detected_result = _get_obj_detection(rgb, img_path, fasterRCNN, args, classes, class_indexs,\
                                                               im_data, im_info, num_boxes, gt_boxes)

                    detected_result_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='pickle')
                    with open(detected_result_path, 'wb') as f:
                       pickle.dump(total_detected_result, f)
	            
                    sim.makeAction([0], [1], [0])
        
                sim.makeAction([0], [0], [-1])
                sim.makeAction([0], [0], [-1])
                for i in range(12):
                    state = sim.getState()[0]
                    rgb = np.array(state.rgb, copy=False)
                    # cv2.imshow('Python RGB', rgb)
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    img_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='img')
                    total_detected_result = _get_obj_detection(rgb, img_path, fasterRCNN, args, classes, class_indexs,\
                                                               im_data, im_info, num_boxes, gt_boxes)

                    detected_result_path = make_save_path(scan_type, viewpoint_type, index_type, data_type='pickle')
                    with open(detected_result_path, 'wb') as f:
                       pickle.dump(total_detected_result, f)
		            
                    sim.makeAction([0], [1], [0])
    
            except Exception as e:
               print('"{}" in "{}" scanId unavailable'.format(viewpoint_type, scan_type))
               print(e)


def get_depth_and_navigable_datas(scantypes, viewpoint_types):
    # Setup simulator
    sim = setup_simulator()
    image_h = 120; image_w = 160

    for scan_type in scan_types:
        print('Processing {}/{}'.format(scan_types.index(scan_type)+1, len(scan_types)))
        viewpoint_types_Ids = viewpoint_types[scan_type]
        for viewpoint_type in viewpoint_types_Ids:
            try:
                index_history = []
                sim.newEpisode([scan_type], [viewpoint_type], [0], [0])
                spatial_depth = np.empty((1, 120*3, 160*12), dtype=np.uint16)
                spatial_n_navigable = np.zeros((1, 3, 12), dtype=np.int32)

                for i in range(12):
                    state = sim.getState()[0]
                    raw_depth = np.squeeze(np.array(state.depth, copy=False))
                    n_navigable = len(state.navigableLocations) - 1
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    y = index_type // 12
                    x = index_type % 12
                    
                    spatial_depth[0, image_h*(2-y):image_h*(3-y), image_w*x:image_w*(x+1)] = raw_depth
                    spatial_n_navigable[0, 2-y, x] = n_navigable
       
                    # cv2.imwrite('depth_ex/{}.png'.format(index_type), np.array(state.depth, copy=False))

                    sim.makeAction([0], [1], [0])

                sim.makeAction([0], [0], [1])
                for i in range(12):
                    state = sim.getState()[0]
                    raw_depth = np.squeeze(np.array(state.depth, copy=False))
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    y = index_type // 12
                    x = index_type % 12

                    spatial_depth[0, image_h*(2-y):image_h*(3-y), image_w*x:image_w*(x+1)] = raw_depth
                    spatial_n_navigable[0, 2-y, x] = n_navigable

                    # cv2.imwrite('depth_ex/{}.png'.format(index_type), np.array(state.depth, copy=False))

                    sim.makeAction([0], [1], [0])

                sim.makeAction([0], [0], [-1])
                sim.makeAction([0], [0], [-1])
                for i in range(12):
                    state = sim.getState()[0]
                    raw_depth = np.squeeze(np.array(state.depth, copy=False))
                    index_type = state.viewIndex
                    index_history.append(index_type)

                    y = index_type // 12
                    x = index_type % 12

                    spatial_depth[0, image_h*(2-y):image_h*(3-y), image_w*x:image_w*(x+1)] = raw_depth
                    spatial_n_navigable[0, 2-y, x] = n_navigable

                    # cv2.imwrite('depth_ex/{}.png'.format(index_type), np.array(state.depth, copy=False))

                    sim.makeAction([0], [1], [0])
                
                file_name = make_save_path(scan_type, viewpoint_type, index_type, data_type='npz_depth')
                np.savez_compressed(file_name, depth=spatial_depth)
                file_name = make_save_path(scan_type, viewpoint_type, index_type, data_type='npz_n_navigable')
                np.savez_compressed(file_name, navigable=spatial_n_navigable)
                # import pdb; pdb.set_trace()
            except Exception as e:
               print('"{}" in "{}" scanId unavailable'.format(viewpoint_type, scan_type))
               print(e)
            
if __name__ == '__main__':
    scan_types, viewpoint_types = read_data_category()
    # scan_types = ['2azQ1b91cZZ']
    # viewpoint_types = {'2azQ1b91cZZ': ['ac3dc08c7a2646b991fda42ccc42bc47']}
    get_obj_detection_datas(scan_types, viewpoint_types)
    get_depth_and_navigable_datas(scan_types, viewpoint_types)




