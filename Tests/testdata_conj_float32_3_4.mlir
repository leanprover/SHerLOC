"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<3x4xf32>
    %6 = "stablehlo.complex"(%2, %5) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    "func.return"(%6) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-0.320381522, -0.114326343, 2.78269958, -1.2944634], [-0.201011941, -0.804424703, 0.361741751, -3.98341298], [-1.02460182, -0.470660597, 0.961570084, -2.37234592]]> : tensor<3x4xf32>}> : () -> tensor<3x4xf32>
    "func.return"(%1) : (tensor<3x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.320381522,0.000000e+00), (-0.114326343,0.000000e+00), (2.78269958,0.000000e+00), (-1.2944634,0.000000e+00)], [(-0.201011941,0.000000e+00), (-0.804424703,0.000000e+00), (0.361741751,0.000000e+00), (-3.98341298,0.000000e+00)], [(-1.02460182,0.000000e+00), (-0.470660597,0.000000e+00), (0.961570084,0.000000e+00), (-2.37234592,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>}> : () -> tensor<3x4xcomplex<f32>>
    "func.return"(%0) : (tensor<3x4xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

