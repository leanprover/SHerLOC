"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf32>, tensor<4x2xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x3xf32>
    %5 = "stablehlo.gather"(%3#0, %3#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>}> : (tensor<4x6xf32>, tensor<4x2xi64>) -> tensor<4x3xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3xf32>, tensor<4x3xf32>) -> ()
    "func.return"(%5) : (tensor<4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf32>, tensor<4x2xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.14172935, 1.45226812, 1.71504426, 0.182875648, 2.24712729, 1.00502801], [3.53438354, 3.140540e+00, -3.78799725, -0.773071169, -7.967960e+00, 2.19334626], [-4.67005682, -0.738040149, 0.920267403, -2.77311182, -2.60643196, -0.117176548], [0.772570908, -3.032180e+00, 1.82749724, -1.43702137, 0.937500596, -4.603724]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1], [1, 2], [2, 3], [3, 2]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
    "func.return"(%1, %2) : (tensor<4x6xf32>, tensor<4x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.45226812, 1.71504426, 0.182875648], [-3.78799725, -0.773071169, -7.967960e+00], [-2.77311182, -2.60643196, -0.117176548], [1.82749724, -1.43702137, 0.937500596]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    "func.return"(%0) : (tensor<4x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

