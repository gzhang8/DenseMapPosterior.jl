import CameraModel
import SLAMData
# import Surfels


import DenseMapPosterior

data_seq_name_w_ext = "livingroom2.klg"
data_seq_name = splitext(data_seq_name_w_ext)[1]


fused_model_fpath = "/tmp/fused_model.ply"

data_dir = joinpath(dirname(dirname(pathof(DenseMapPosterior))), "data", "livingroom2")

data_file = joinpath(data_dir, data_seq_name_w_ext)
cam = CameraModel.Aug_ICL_NUIM_CamParam()
rgbd_reader = SLAMData.KlgReader(data_file, cam=cam);

# traj_path = joinpath(data_dir, "m1.log")   # -14.06029388
traj_path = joinpath(data_dir, "m2.log")   # -37.356114
# traj_path = joinpath(data_dir, "m3.log")   # -48.314537
# traj_path = joinpath(data_dir, "m4.log")   # -982.4293978
# traj_path = joinpath(data_dir, "m5.log")   # -2252.5785928

est_name = splitext(basename(traj_path))[1]
traj = SLAMData.load_redwood_traj(traj_path)


# ------------------- fuse model -------------------------------------------
# create opengl context

# viewer_ = Surfels.viewer()

DenseMapPosterior.fuse_RGBD_to_3D_model(
    rgbd_reader,
    traj.poses, #id2Twc
    rgbd_reader.cam,
    fused_model_fpath
)

# -------------------- eval DMP ---------------------------------------------



dmp = DenseMapPosterior.DMP_RGBD(
    rgbd_reader,
    fused_model_fpath,
    traj.poses,
    cam
)

println("DMP of $est_name on $(data_seq_name) is: $dmp")
