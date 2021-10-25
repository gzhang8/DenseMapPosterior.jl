import ElasticFusion
import Surfels
import Pangolin
import SLAMData
import CameraModel


function fuse_RGBD_to_3D_model(
    rgbd_reader::T,
    id2Twc, # traj
    cam::CameraModel.RgbdCamParams,
    output_ply_fpath::String
) where T <: SLAMData.RgbdReader

    # hidden_window = Surfels.create_hidden_gl_window(cam.width, cam.height)
    window_name = "EFusion Window"
    Pangolin.create_window(window_name, cam.width, cam.height)

    ef = ElasticFusion.Fusion(
        w=cam.width,
        h=cam.height,
        fx=cam.fx,
        fy=cam.fy,
        cx=cam.cx,
        cy=cam.cy,
        timeDelta = 200,
        countThresh = 35000,
        errThresh = 5e-05,
        covThresh = 1e-05,
        closeLoops = false,
        iclnuim = false,
        reloc = false,
        photoThresh = 115,
        confidence = 10,
        depthCut = 5,
        icpThresh = 10,
        fastOdom = false,
        fernThresh = 0.3095,
        so3 = true,
        frameToFrameRGB = false,
        fileName = "fsd_2021sep"
    )

    frame_ids = sort!(collect(keys(id2Twc)))
    @assert frame_ids[1] == 0

    for f_id = frame_ids

        # TODO: how to get timestamp from id
        time_stamp = Int64(f_id)

        # pose = convert(Matrix{Float32}, inv(all_Twc0[f_id]))
        pose = convert(Matrix{Float32}, id2Twc[f_id])

        img, depth16 = SLAMData.readRGBD(rgbd_reader, id=time_stamp)

        # note depth_factor only support 1000.0 for now
        @assert cam.depth_factor == 1000.0

        ElasticFusion.process_frame!(
            ef, img, depth16, time_stamp, pose=pose, depth_factor=1000.0
        ) 

        # function process_frame!(
        #     ef::ElasticFusion,
        #     image::Matrix{RGB{Normed{UInt8,8}}},
        #     depth::Matrix{UInt16},
        #     timestamp::Int64,
        #     pose::Matrix{Float32};
        #     weightMultiplier::Union{Float32, Float64}
        #     bootstrap::Bool
        #     )

        # Pangolin.activate(surfel_viewer.d_cam, surfel_viewer.s_cam)
        # glClearColor(0.05 , 0.05 , 0.3 , 0.0);
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        # # end
        # mvp = Pangolin.get_projection_view_mat(surfel_viewer.s_cam)
        # # mvp32 = convert(Matrix{Float32}, mvp)

        # ElasticFusion.render(
        #     ef,
        #     mvp,
        #     threshold=10,#::Union{Float64, Float32},10
        #     drawUnstable=true,#//bool
        #     drawNormals=false,#, //bool
        #     drawColors=true,#,  //bool
        #     drawPoints=true,#,  //bool
        #     drawWindow=false,#,  //bool
        #     drawTimes=false,#,   //bool
        #     time=time_stamp,
        #     timeDelta=200
        # )
        # Pangolin.finish_frame()
    end

    ElasticFusion.save_ply(ef, output_ply_fpath)
    finalize(ef)
    Pangolin.destroy_window(window_name)
end