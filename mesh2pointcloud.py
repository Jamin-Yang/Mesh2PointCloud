import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
# load mesh


class Mesh2PointCloud:
    def __init__(self, horizontal_res = 1024, vertical_res = 128, horizontal_fov = 360,
                 vertical_fov = 45, add_noise = False, noise_std = 0.01):

        self.horizontal_res = horizontal_res
        self.vertical_res = vertical_res
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.vertical_ray_angle = np.array([np.pi/180*(-self.vertical_fov/2 + self.vertical_fov/(self.vertical_res-1)*i) 
                                            for i in range(self.vertical_res)]) # (-fov, fov)
        self.horizontal_ray_angle = np.array([np.pi/180*(self.horizontal_fov/self.horizontal_res*i) 
                                              for i in range(horizontal_res)])  # (0, 360)
        # noise
        self.add_noise = add_noise
        self.noise_std = noise_std

    def create_scene(self, mesh):
        # convert to t.geometry
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        # create raycasting scene
        scene = o3d.t.geometry.RaycastingScene()

        # add mesh to scene
        mesh_id = scene.add_triangles(t_mesh)
        return scene
    


    def cast_rays(self, scene, radar_position):
        # construct scene and initialize a point cloud
        pointcloud = o3d.geometry.PointCloud()

        for v in self.vertical_ray_angle:
            if np.abs(np.sin(v)) > np.sin(np.pi/180*12):
                continue
            for h in self.horizontal_ray_angle:
                # create normalized ray direction
                view = np.array([np.sin(h)*np.cos(v), np.sin(v), np.cos(h)*np.cos(v)])
                # if view dot radar_position < 0, then the ray is not casted
                if np.dot(view, -radar_position) < np.cos(np.pi/180*12):
                    continue
                # compose ray
                ray = o3d.core.Tensor([[radar_position[0], radar_position[1], radar_position[2],
                                         view[0], view[1], view[2]]], dtype=o3d.core.Dtype.Float32)
                # cast ray
                ans = scene.cast_rays(ray)
                # determine if there is an intersection
                if ans['t_hit'].numpy() == np.inf:
                    continue
                else:
                    # intersection point
                    if self.add_noise is False:
                        distance = ans['t_hit'].numpy()
                    else:
                        distance = ans['t_hit'].numpy() + np.random.normal(0, self.noise_std)
                    # compute position and normals
                    position = radar_position + view * distance
                    position = position.reshape(1,3)
                    normals = ans['primitive_normals'].numpy()
                    normals = normals.reshape(1,3)
                    # if the point cloud is empty, initialize it
                    if len(pointcloud.points) == 0:
                        pointcloud.points = o3d.utility.Vector3dVector(position)
                        pointcloud.normals = o3d.utility.Vector3dVector(normals)
                    else:
                        # append to point cloud
                        pointcloud.points.extend(o3d.utility.Vector3dVector(position))
                        pointcloud.normals.extend(o3d.utility.Vector3dVector(normals))
        return pointcloud
    

    def forward(self, mesh, radar_position, meshID, view_angle):
        scene = self.create_scene(mesh)
        pointcloud = self.cast_rays(scene,radar_position)
        # write point cloud to current folder
        # get current path
        current_path = os.path.dirname(__file__)
        # create folder name 'xyznxnynz'
        sub_folder_name0 = 'pointcloud'
        sub_folder_name1 = '{:04d}'.format(meshID)
        sub_folder_name2 = 'pc_xyznxnynz'
        # make directory if 'pointcloud/meshID/pc_xyznxnynz' not exist
        path_name = os.path.join(current_path, sub_folder_name0, sub_folder_name1, sub_folder_name2)
        if os.path.exists(path_name) is False:
            os.makedirs(path_name, exist_ok=True)
        # create file name '0i0.ply'
        file_name = '{:03d}.ply'.format(view_angle*10)
        # concatenate path
        desti_file_path = os.path.join(path_name, file_name)
        o3d.io.write_point_cloud(desti_file_path, pointcloud)



if __name__ == '__main__':
    # in the mesh obj, the unit is 1.7m, thus the position of camera in scene is [0,0,5]/1.7
    real_physical_res_ratio = 1.7
    horizontal_res = 1024
    vertical_res = 128
    horizontal_fov = 360
    vertical_fov = 45
    add_noise = True
    noise_std = 0.01/real_physical_res_ratio

    # radar position
    radar_to_mesh_distance = 5/real_physical_res_ratio

    # create mesh2pc object
    mesh2pc = Mesh2PointCloud(horizontal_res, vertical_res, horizontal_fov, vertical_fov, add_noise, noise_std)
    for i in tqdm(range(526)):
        # load mesh
        source_file_path = 'THuman2.0_Release_copy'
        desti_file_path = 'pointcloud'
        file_name = '{:04d}'.format(i)
        mesh_path = os.path.join(source_file_path, file_name, file_name+'.obj')
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        for angle in tqdm(range(36)):
            # create radar position
            theta = np.pi/180*10*angle
            radar_position = np.array([np.sin(theta),0,np.cos(theta)]) * radar_to_mesh_distance
            # forward and get 36 point clouds with different angles
            mesh2pc.forward(mesh, radar_position, i, angle)