"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3x3xf32>, tensor<2x3xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x3x2xf32>
    %5 = "stablehlo.gather"(%3#0, %3#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 2>}> : (tensor<2x3x3xf32>, tensor<2x3xi64>) -> tensor<2x3x2xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3x2xf32>, tensor<2x3x2xf32>) -> ()
    "func.return"(%5) : (tensor<2x3x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3x3xf32>, tensor<2x3xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[0.0618097708, 0.232754916, 0.668427229], [4.27152205, -3.72517085, -3.77432418], [1.83894026, -4.33732605, -1.31484962]], [[-3.41498137, 0.911645174, 0.27581653], [-1.31400931, 0.393603295, 2.46858478], [-7.053160e-01, 2.62368941, 3.240237]]]> : tensor<2x3x3xf32>}> : () -> tensor<2x3x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 0], [1, 2, 1]]> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    "func.return"(%1, %2) : (tensor<2x3x3xf32>, tensor<2x3xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.0618097708, 0.232754916], [4.27152205, -3.72517085], [1.83894026, -4.33732605]], [[0.911645174, 0.27581653], [0.393603295, 2.46858478], [2.62368941, 3.240237]]]> : tensor<2x3x2xf32>}> : () -> tensor<2x3x2xf32>
    "func.return"(%0) : (tensor<2x3x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

