"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<7xf32>, tensor<1xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %5 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xi64>) -> tensor<1xi64>
    %6 = "stablehlo.reshape"(%5) : (tensor<1xi64>) -> tensor<i64>
    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %8 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %9 = "stablehlo.constant"() <{value = dense<7> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.add"(%6, %9) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %11 = "stablehlo.select"(%8, %10, %6) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %12 = "stablehlo.dynamic_slice"(%3#0, %11) <{slice_sizes = array<i64: 3>}> : (tensor<7xf32>, tensor<i64>) -> tensor<3xf32>
    "stablehlo.custom_call"(%12, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%12) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<7xf32>, tensor<1xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[2.28955293, 1.20324981, -1.75449681, 0.685895442, -0.487509489, -1.3147161, 1.81431377]> : tensor<7xf32>}> : () -> tensor<7xf32>
    %2 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi64>}> : () -> tensor<1xi64>
    "func.return"(%1, %2) : (tensor<7xf32>, tensor<1xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-0.487509489, -1.3147161, 1.81431377]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

