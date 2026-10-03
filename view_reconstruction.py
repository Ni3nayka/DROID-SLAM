import argparse
import open3d as o3d

from reconstruction_view import build_reconstruction, configure_rendering, create_camera_actor

def view_reconstruction(filename: str, filter_thresh = 0.005, filter_count=2):
    point_cloud, pose_mats = build_reconstruction(filename, filter_thresh, filter_count)

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    configure_rendering(vis)

    vis.add_geometry(point_cloud)

    ### add camera actor ###
    for pose in pose_mats:
        cam_actor = create_camera_actor(pose)
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    args = parser.parse_args()

    view_reconstruction(args.filename, args.filter_threshold, args.filter_count)
