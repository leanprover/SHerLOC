"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x6x3xf32>, tensor<2x3xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x3x3xf32>
    %5 = "stablehlo.gather"(%3#0, %3#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 3>}> : (tensor<2x6x3xf32>, tensor<2x3xi64>) -> tensor<2x3x3xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3x3xf32>, tensor<2x3x3xf32>) -> ()
    "func.return"(%5) : (tensor<2x3x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x6x3xf32>, tensor<2x3xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-2.35533047, -1.08525527, -0.50400424], [-4.75908756, -3.10147309, 5.32185507], [5.54974031, 3.52409244, 3.53000522], [1.45906162, -5.08333111, -4.1355195], [0.768421888, -1.49637628, -2.71538854], [0.763425469, -4.3264246, -0.999291121]], [[-7.463970e-01, 4.11092663, 3.20970058], [5.62080479, -2.63781524, -1.27055144], [3.88063931, -1.46174836, 0.184568912], [1.9948988, 3.40338898, -4.73411942], [4.25097942, 2.53524303, 5.71731091], [-0.358707964, -2.31571078, -2.11293054]]]> : tensor<2x6x3xf32>}> : () -> tensor<2x6x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 0], [1, 2, 0]]> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    "func.return"(%1, %2) : (tensor<2x6x3xf32>, tensor<2x3xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-4.75908756, -3.10147309, 5.32185507], [5.54974031, 3.52409244, 3.53000522], [1.45906162, -5.08333111, -4.1355195]], [[3.88063931, -1.46174836, 0.184568912], [1.9948988, 3.40338898, -4.73411942], [4.25097942, 2.53524303, 5.71731091]]]> : tensor<2x3x3xf32>}> : () -> tensor<2x3x3xf32>
    "func.return"(%0) : (tensor<2x3x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

