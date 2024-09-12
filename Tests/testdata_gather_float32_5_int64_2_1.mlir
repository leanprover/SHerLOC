"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5xf32>, tensor<2x1xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xf32>
    %5 = "stablehlo.gather"(%3#0, %3#1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<5xf32>, tensor<2x1xi64>) -> tensor<2xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf32>, tensor<2xf32>) -> ()
    "func.return"(%5) : (tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5xf32>, tensor<2x1xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[1.27891302, 0.396668941, 2.52560949, 1.63635993, 2.47598505]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0], [2]]> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    "func.return"(%1, %2) : (tensor<5xf32>, tensor<2x1xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.27891302, 2.52560949]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%0) : (tensor<2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

