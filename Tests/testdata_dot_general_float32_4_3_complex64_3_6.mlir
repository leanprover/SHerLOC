"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xf32>, tensor<3x6xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f32>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xf32>) -> tensor<4x3xcomplex<f32>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f32>>) -> tensor<3x6xcomplex<f32>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xf32>, tensor<3x6xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.021900e+00, 2.94948554, 0.246437892], [0.462476343, -2.05846119, 0.508504629], [-0.654622793, 0.00105758384, 2.94362497], [-1.19875145, -1.24357963, -5.457330e+00]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[(-5.04119205,3.43552709), (0.838695228,0.68958962), (6.17671442,-1.46617091), (1.72184861,1.71176279), (5.21125412,-0.894076228), (0.917864799,3.1008935)], [(-1.64409041,3.11131096), (0.275597364,-0.433356345), (-1.63204396,-4.27285433), (1.3342526,-1.51156056), (0.28396371,-6.35333681), (0.154464349,0.742328703)], [(1.18377459,1.4673413), (1.78410256,-2.67910743), (-2.90774846,-8.256880e-03), (-6.0200262,-0.539057791), (2.85339594,-0.886770367), (-1.16269314,-7.29733467)]]> : tensor<3x6xcomplex<f32>>}> : () -> tensor<3x6xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<4x3xf32>, tensor<3x6xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-14.7502804,16.4846668), (2.94829893,-0.544130564), (6.95842934,-15.5692081), (5.93320178,-1.13015699), (12.0773668,-20.7653427), (2.02488947,6.66084384)], [(1.65481925,-4.06951284), (0.727794647,-0.151372433), (4.737480e+00,8.11323642), (-5.01140404,3.62902522), (3.27651882,12.2136812), (-0.484702975,-3.80469322)], [(6.78292847,2.07361865), (4.70299149,-8.33816719), (-12.6044655,0.930964827), (-18.8464489,-2.70894146), (4.98822212,-2.03175592), (-4.02322435,-23.5097485)], [(1.62744427,-15.9952717), (-11.0845509,14.3330412), (10.4937744,7.11626959), (29.1299534,2.76958418), (-22.1720543,13.8120537), (5.052820e+00,35.1836205)]]> : tensor<4x6xcomplex<f32>>}> : () -> tensor<4x6xcomplex<f32>>
    "func.return"(%0) : (tensor<4x6xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

