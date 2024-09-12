"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<bf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<bf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<bf16>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<bf16>) -> tensor<bf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<bf16>, tensor<bf16>) -> ()
    "func.return"(%4) : (tensor<bf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<bf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<1.046880e+00> : tensor<bf16>}> : () -> tensor<bf16>
    "func.return"(%1) : (tensor<bf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<bf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<1.046880e+00> : tensor<bf16>}> : () -> tensor<bf16>
    "func.return"(%0) : (tensor<bf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

