"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf32>, tensor<1xui32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<1xf32>
    %5 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xui32>) -> tensor<1xui32>
    %6 = "stablehlo.reshape"(%5) : (tensor<1xui32>) -> tensor<ui32>
    %7 = "stablehlo.dynamic_slice"(%3#0, %6) <{slice_sizes = array<i64: 1>}> : (tensor<3xf32>, tensor<ui32>) -> tensor<1xf32>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xf32>, tensor<1xf32>) -> ()
    "func.return"(%7) : (tensor<1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf32>, tensor<1xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[1.38056946, 4.05531645, 5.88274193]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xui32>}> : () -> tensor<1xui32>
    "func.return"(%1, %2) : (tensor<3xf32>, tensor<1xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<4.05531645> : tensor<1xf32>}> : () -> tensor<1xf32>
    "func.return"(%0) : (tensor<1xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

