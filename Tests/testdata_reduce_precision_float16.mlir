"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<f16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<f16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<f16>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<f16>) -> tensor<f16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<f16>, tensor<f16>) -> ()
    "func.return"(%4) : (tensor<f16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<5.087890e-01> : tensor<f16>}> : () -> tensor<f16>
    "func.return"(%1) : (tensor<f16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<f16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.087890e-01> : tensor<f16>}> : () -> tensor<f16>
    "func.return"(%0) : (tensor<f16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

