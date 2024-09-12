"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf64>, tensor<1xf64>, tensor<1xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3xf64>
    %6 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xi64>) -> tensor<1xi64>
    %7 = "stablehlo.reshape"(%6) : (tensor<1xi64>) -> tensor<i64>
    %8 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %9 = "stablehlo.compare"(%7, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %10 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %11 = "stablehlo.add"(%7, %10) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %12 = "stablehlo.select"(%9, %11, %7) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %13 = "stablehlo.dynamic_update_slice"(%4#0, %4#1, %12) : (tensor<3xf64>, tensor<1xf64>, tensor<i64>) -> tensor<3xf64>
    "stablehlo.custom_call"(%13, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf64>, tensor<3xf64>) -> ()
    "func.return"(%13) : (tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf64>, tensor<1xf64>, tensor<1xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[3.4330394722198463, -4.0526859674631508, -3.7716480953971332]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %2 = "stablehlo.constant"() <{value = dense<3.7596441190639984> : tensor<1xf64>}> : () -> tensor<1xf64>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    "func.return"(%1, %2, %3) : (tensor<3xf64>, tensor<1xf64>, tensor<1xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[3.4330394722198463, 3.7596441190639984, -3.7716480953971332]> : tensor<3xf64>}> : () -> tensor<3xf64>
    "func.return"(%0) : (tensor<3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

