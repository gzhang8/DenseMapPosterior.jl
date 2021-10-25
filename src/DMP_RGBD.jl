import CUDA
using ModernGL

# 6 calculate posterior
# get likelihood, CPU version
function depth_nll(depth_predict::Matrix, depth_real::Matrix)

    n_rows, n_cols = size(depth_predict)

    nll = 0.0
    vaild_count = 0
    for c_idx = 1:n_cols
        for r_idx = 1:n_rows
            real_d_value = depth_real[r_idx, c_idx]
            predict_d_value = depth_predict[r_idx, c_idx]
            if (predict_d_value != 0.0) && (real_d_value != 0.0)
                diff = predict_d_value - real_d_value
                # @show diff
                nll = nll - diff^2
                vaild_count += 1
            end
        end
    end
    return nll / max(1, vaild_count)
end

"""
for best performance, set tmp from a outside array
This is ll not nll, the negative is in ll equation
"""
function depth_nll_cuda(frame_buffer::Surfels.GLFrameBuffer,
                        depth_real_f32_cu::CUDA.CuArray{Float32, 2};
                        tmp::CUDA.CuArray{Float32, 2}=similar(depth_real_f32_cu))
    w = frame_buffer.cam.width
    h = frame_buffer.cam.height
    Surfels.glCheckError()
    glCopyImageSubData(frame_buffer.depth_render_buf, GL_TEXTURE_2D, 0, 0, 0, 0,
                       frame_buffer.depth_2nd_tex, GL_TEXTURE_2D, 0, 0, 0, 0,
                       w, h, 1);
    Surfels.glCheckError()

    # # ccall to cuda calculate
    # using CUDA
    # depth_real_f32 = convert(Matrix{Float32}, depth_real_uint) ./ 1000.0
    # depth_real_f32_cu = cu(depth_real_f32)
    # out_log_likelihood_cu = cu(depth_real_f32)
    # a = proj_mat[3, 3]
    # b = proj_mat[3, 4]
    # c = proj_mat[4, 3]
    # void cu_depth_log_likelihood(float* depth_real_ptr,
    #                                void** depth_predict_tex_cuda_res_ptr,
    #                                int width,
    #                                int height,
    #                                double a,
    #                                double b,
    #                                double c,
    #                                float max_depth,
    #                                float* out_log_likelihood);
    out_log_likelihood_cu = tmp
    ccall((:cu_depth_log_likelihood, :libsurfels),
          Cvoid,
    	  (CUDA.CuPtr{Cfloat}, Ptr{Ptr{Cvoid}}, Cint, Cint, Cdouble, Cdouble,
    	  Cdouble, Cfloat, Cfloat, Cint, Cint, CUDA.CuPtr{Cfloat}),
    	  depth_real_f32_cu.storage.buffer.ptr,
    	  Base.unsafe_convert(Ptr{Ptr{Cvoid}}, frame_buffer.depth_cuda_res),
    	  w,
    	  h,
    	  frame_buffer.a,
    	  frame_buffer.b,
    	  frame_buffer.c,
    	  frame_buffer.min_depth,
    	  frame_buffer.max_depth,
    	  16, # thread block size: 16 x 16
    	  Int32(0),
    	  out_log_likelihood_cu.storage.buffer.ptr)

    # out = collect(out_log_likelihood_cu)
    # imshow(Gray.(out))

    cu_nll = - sum(out_log_likelihood_cu) #/ sum(out_log_likelihood_cu .!= 0.0)
    cu_count = sum(out_log_likelihood_cu .!= 0.0)
    return cu_nll / max(1, cu_count)
end

function DMP_RGBD(
    rgbd_reader::T,
    model_fpath::String,
    id2Twc, # traj
    cam::CameraModel.RgbdCamParams;
    min_depth::Float64 = 0.3,
    max_depth::Float64 = 15.0,
    confidence_thres::Float64=0.1,
    max_id::Int64=100000000000000,
    tmp::CUDA.CuArray{Float32, 2}=CUDA.CuArray{Float32, 2}(
        undef, Int64(cam.height), Int64(cam.width)
    )
) where T <: SLAMData.RgbdReader

    hidden_window = Surfels.create_hidden_gl_window(cam.width, cam.height)

    program::ModernGL.GLuint = Surfels.create_gl_render_program()

    frame_buffer::Surfels.GLFrameBuffer = Surfels.GLFrameBuffer(
        cam.width, cam.height, cam, min_depth, max_depth, use_texture = true
    )

    # load GlModel
    model::Surfels.GlSurfelsData = Surfels.read_surfels_file2gl(model_fpath)

    nll = 0.0
    count = 0
    Surfels.glCheckError()

    ids = sort(collect(keys(id2Twc)))

    for frame_id = ids
        if frame_id > max_id
            continue
        end
        #render to get predicted observation
        Surfels.render(
            [model],
            id2Twc[frame_id], # Twc
            confidence_thres,
            1,
            200000,
            program,
            frame_buffer
        )
        Surfels.glCheckError()

        # real observation
        # depth_real_f32_cu = frames_d_cu[frame_id]
        img, depth_real_uint = SLAMData.readRGBD(rgbd_reader, id=frame_id)
        depth_real_f = convert(Matrix{Float32}, depth_real_uint) ./ 1000.0
        depth_real_f32_cu = CUDA.cu(depth_real_f)

        nll += depth_nll_cuda(frame_buffer, depth_real_f32_cu, tmp=tmp)

        Surfels.glCheckError()
        count += 1
    end
    # TODO destroy program
    finalize(frame_buffer)
    finalize(model)
    finalize(hidden_window)
    
    return nll #/ Float64(count)
end
