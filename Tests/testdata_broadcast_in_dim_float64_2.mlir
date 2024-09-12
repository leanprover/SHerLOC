"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2xf64>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf64>, tensor<2xf64>) -> ()
    "func.return"(%2) : (tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-1.7208777278819087, 2.0499863811108221]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1) : (tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.7208777278819087, 2.0499863811108221]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%0) : (tensor<2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

