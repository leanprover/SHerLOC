"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<1x2xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2xf64>
    %4 = "stablehlo.reshape"(%2) : (tensor<1x2xf64>) -> tensor<2xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf64>, tensor<2xf64>) -> ()
    "func.return"(%4) : (tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.4748586358361, 0.73655167448850767]]> : tensor<1x2xf64>}> : () -> tensor<1x2xf64>
    "func.return"(%1) : (tensor<1x2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.4748586358361, 0.73655167448850767]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%0) : (tensor<2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

