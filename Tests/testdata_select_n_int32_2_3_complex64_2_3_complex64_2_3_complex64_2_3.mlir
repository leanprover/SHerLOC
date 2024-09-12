"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f32>>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    "func.return"(%14) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 2, 2], [2, 1, 2]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[(3.93428254,0.910755396), (-0.824884116,6.81380701), (0.620987177,-2.56805634)], [(3.92014217,-4.66776276), (-3.18807101,0.409840196), (-4.76146793,0.942774832)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "stablehlo.constant"() <{value = dense<[[(-3.53006744,2.75585723), (6.491930e+00,-2.32900286), (1.81959879,2.03441906)], [(0.980624616,1.46047854), (-2.48105192,1.55319786), (-2.05409145,1.94431758)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<[[(-0.700041056,-2.54219437), (-3.90275693,-2.60665703), (-1.46083939,0.337038815)], [(-2.17983437,1.1594137), (-0.577557862,0.929602205), (2.11863852,-0.408816457)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.700041056,-2.54219437), (-3.90275693,-2.60665703), (-1.46083939,0.337038815)], [(-2.17983437,1.1594137), (-2.48105192,1.55319786), (2.11863852,-0.408816457)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%0) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

