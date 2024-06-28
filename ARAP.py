import csv
import time
import copy
import open3d as o3d
import numpy as np
from scipy import spatial


def closest_point(source_p, target_p):
    dist, indexes = spatial.KDTree(target_p).query(source_p)
    return indexes


def import_cvs(file_path):
    with open(file_path, 'r') as file:
        points = csv.reader(file)
        values = []
        for row in points:
            values.append(row)
    return values

def ARAP(source_points, target_points, source_handles, target_handles)

    # Start timer
    # The source would be all the SIFT points

    start = time.time()
    print("Reading meshes ...")
    source = o3d.io.read_point_cloud("data/Test1/source.ply")

    print("Reading SIFT points  ...")
    source_points = import_cvs('data/Test1/source_SIFT_filtered.csv')
    target_points = import_cvs('data/Test1/target_SIFT_filtered.csv')

    # Creating mesh with poisson surface reconstruction
    # Took similar time to nested loop (without removing boundaries)
    # 10x faster than Delaunay
    print("Creating mesh triangles ...")

    # Equal z axis of last row to 1 for aligned bounding box to work
    # Comment if a 3D point cloud is given
    last_row = np.asarray(source.points)
    last_row[-1:, -1:] = 1

    # Estimate normals for mesh reconstruction
    source.estimate_normals()

    # Poisson Surface Reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(source, depth=9)[0]
    # Delete points or vertices outside the edge of original point cloud
    bbox = source.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Simplifying mesh to 70,000 triangles to improve speed performance while maintaining features
    # source = source.simplify_quadric_decimation(70000)

    # Extract the closest point on mesh that matches the SIFT points to be deformed
    print("Finding closest point ...")
    match_idx = closest_point(source_points, np.asarray(mesh.vertices))

    # Delete repeated SIFT points or points that match same vertices
    print("Creating new array with target's points ...")
    modified_points = copy.copy(mesh.vertices)
    clear_idx = []
    points = []
    for (i, target_points) in zip(match_idx, target_points):
        modified_points[i] = target_points
        if i not in clear_idx:
            clear_idx.append(i)
            points.append(np.asarray(modified_points[i]))

    print("Performing ARAP ...")
    # o3d.visualization.draw_geometries([mesh])
    # Remove incorrect artifacts for better consistency
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # ARAP
    # Idx of points to be deformed
    constraint_ids = o3d.utility.IntVector(clear_idx)
    # Final position of deformed points
    constraint_pos = o3d.utility.Vector3dVector(points)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids,
                                                      constraint_pos,
                                                      max_iter=50)

    # Compute normals of output mesh
    mesh_prime.compute_vertex_normals()
    # Compute colours of output mesh
    mesh_prime.vertex_colors = mesh.vertex_colors

    # Equal all z values to 0 for 2D mesh
    mesh_2d = np.asarray(mesh_prime.vertices)
    mesh_2d[:, -1:] = 0
    #print(mesh_2d)
    #print(np.asarray(mesh.vertices))
    #print(np.asarray(mesh_prime.vertices))
    # Finish timer for code
    end = time.time()
    # Visualize mesh
    #o3d.visualization.draw_geometries([mesh])
    #o3d.visualization.draw_geometries([mesh_prime])
    # Print time it took for code to run
    print('Completed in...', end - start)

    # Create an image
    #vis = o3d.visualization.Visualizer()
    #vis.create_window(visible=False)  # works for me with False, on some systems needs to be true
    #vis.add_geometry(mesh_prime)
    #vis.update_geometry(mesh_prime)
    #vis.poll_events()
    #vis.update_renderer()
    #vis.capture_screen_image('miage.png')
    #vis.destroy_window()

    # Save mesh to folder
    o3d.io.write_triangle_mesh("data/Test1/output_ARAP.ply", mesh_prime)
    o3d.io.write_triangle_mesh("data/Test1/output_ARAP2.ply", mesh)
    o3d.io.write_triangle_mesh("data/Test1/output_ARAP.stl", mesh_prime)

    return mesh_prime