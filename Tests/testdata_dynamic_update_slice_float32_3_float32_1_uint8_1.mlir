"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf32>, tensor<1xf32>, tensor<1xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %6 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<1xui8>) -> tensor<1xui8>
    %7 = "stablehlo.reshape"(%6) : (tensor<1xui8>) -> tensor<ui8>
    %8 = "stablehlo.dynamic_update_slice"(%4#0, %4#1, %7) : (tensor<3xf32>, tensor<1xf32>, tensor<ui8>) -> tensor<3xf32>
    "stablehlo.custom_call"(%8, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%8) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf32>, tensor<1xf32>, tensor<1xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-3.38123536, 5.99582672, 0.870595455]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.219151393> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xui8>}> : () -> tensor<1xui8>
    "func.return"(%1, %2, %3) : (tensor<3xf32>, tensor<1xf32>, tensor<1xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-3.38123536, 0.219151393, 0.870595455]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

