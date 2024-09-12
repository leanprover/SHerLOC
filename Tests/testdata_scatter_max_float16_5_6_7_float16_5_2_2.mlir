"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2x2xi64>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xBE3D6FC5594108A53BC4B8399844EA40174198C609443BBA843B2146DF45B0C31D420B405BC0393E8BC40F36A444B840F84383B971C476C43138331F1BB8A13472B821C6FF38CA33D44025C507BFC6431DBFC73C3CC163B5C9BF56C20D428544F246F046B2BF58BEE4459DC780C4AB3E2B44D2C649BB752AF4BD74C1FDBC42466A391C41CE4001413BBD36426140FC4591B5B8B990C53E3FB0C151C4EC3D0F38A6427C41EB3C96C48F3E2039D2BF5A2FB43D6F3722462C43A23EECB21AC29B442946974028C18B413CAF04C08440FB441DC0E53C0EC13439253A39C5BB44BEBB1A400CC5933A62309EBDC942CCBD07BC8D38583E69435E43A445733EE6356144403BA54400413E4137B71848DBC4823797C2973AB0B7F13D744041C534C22ABF7344FBBE8246BDC1483E81C4793C7C322D4563B8A6C38B3FFF40183FDB3F34442DBD77BD73C495411A37A2B36A44E5C0804406405FB43741C44069C202BC563FB5464FBC613A14AFC5C469BC38C1E4C52B416CC29A3B95BDA4C365B9AC4119C437C2ED42B5C5F8432FC2824019BC2EC4083CDBC1D93F4343ADBBF8BD913E31412FBFD740"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3.951170e+00, -3.437500e+00], [-5.395510e-01, -6.040040e-01]], [[-1.125000e+00, 6.958000e-01], [6.372070e-01, 2.283200e+00]], [[2.208980e+00, -2.777340e+00], [3.138670e+00, 1.079100e+00]], [[-3.955080e+00, -3.894530e+00], [-1.390630e+00, 1.407230e+00]], [[1.396480e+00, 4.308590e+00], [2.027340e+00, -2.533200e+00]]]> : tensor<5x2x2xf16>}> : () -> tensor<5x2x2xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<5x2x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xBE3DE7C3594108A53BC4B8399844EA401741D5B809443BBA843B2146DF45B0C31D420B405BC0393E8BC40F36A444B840F84383B971C476C43138331F1BB8A13472B821C6FF38CA33D44025C507BFC6431DBFC73C3CC163B5C9BF56C20D428544F246F046B2BF9140E4459DC780C4AB3E2B44D2C649BB9139F4BD74C1FDBC42466A391C41CE4001413BBD36426140FC4591B5B8B990C53E3FB0C151C4EC3D0F38A6427C41EB3C96C48F3E6B40D2BF5A2FB43D6F3722462C43A23E513C1AC29B442946974028C18B413CAF04C08440FB441DC0E53C0EC13439253A39C5BB44BEBB47420CC5933A62309EBDC942CCBD07BC8D38583E69435E43A445733EE6356144403BA54400413E4137B71848DBC4A13D97C2973AB0B7F13D744041C534C22ABF7344FBBE8246BDC1483E81C4793C7C322D4563B890BD8B3FFF40183FDB3F34442DBD77BD73C495411A37A2B36A44E5C0804406405FB43741C44069C202BC563FB5464FBC613A14AFC5C469BC38C1E4C52B414F449A3B95BDA4C365B9AC4119C437C2ED42B5C5F8430E40824019BC2EC4083CDBC1D93F4343ADBBF8BD913E31412FBFD740"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

