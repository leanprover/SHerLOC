"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf32>, tensor<2xf32>, tensor<1xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %6 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xi64>) -> tensor<1xi64>
    %7 = "stablehlo.reshape"(%6) : (tensor<1xi64>) -> tensor<i64>
    %8 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %9 = "stablehlo.compare"(%7, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %10 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %11 = "stablehlo.add"(%7, %10) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %12 = "stablehlo.select"(%9, %11, %7) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %13 = "stablehlo.dynamic_update_slice"(%4#0, %4#1, %12) : (tensor<3xf32>, tensor<2xf32>, tensor<i64>) -> tensor<3xf32>
    "stablehlo.custom_call"(%13, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%13) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf32>, tensor<2xf32>, tensor<1xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[0.523907483, 4.45913839, -3.29833961]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[3.05733466, 0.742306352]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %3 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi64>}> : () -> tensor<1xi64>
    "func.return"(%1, %2, %3) : (tensor<3xf32>, tensor<2xf32>, tensor<1xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.523907483, 3.05733466, 0.742306352]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

